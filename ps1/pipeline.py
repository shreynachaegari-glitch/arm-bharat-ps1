"""W10: the offline decision pipeline with per-stage timers (v4 section 4).

photo -> quality gate -> input gate -> vision .pte -> decision engine -> reviewed card -> Screen 1
      -> (optional) buffered Llama explanation -> checker -> Screen 2

Screen 1 never waits for Llama. Every stage is timed, so the same code yields the G2 latency table on
the Pi. Usage (x86 dev, draft cards):
  python -m ps1.pipeline --pte runs/v4_f0/export/tomato_effnet_b0_t2_fp32_xnnpack.pte \\
      --decision runs/v4_f0/export/decision.json --allow-drafts --image photo.jpg [--llama]
"""
import argparse
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ps1 import quality_gate
from ps1.cards import (
    CHEMICAL_CONTROL_LINE,
    CONTACT_SOURCE,
    Card,
    load_cards,
    sources_for,
)
from ps1.contracts import (
    AbstainReason,
    Advice,
    Finding,
    Lang,
    Request,
    check_advice_matches,
)
from ps1.decision import DecisionConfig, decide
from ps1.llm.explain import LlamaConfig, build_prompt, explain
from ps1.vision.runtime import VisionModel, load_vision

# Fixed, checker-clean texts for abstentions (no disease is named; Llama is never called for these).
ABSTAIN_TEXT = {
    AbstainReason.POOR_QUALITY: "The photo is not clear enough. Please take another photo of one leaf in good light.",
    AbstainReason.NOT_A_PLANT: "This does not look like a plant leaf. Please take a photo of one tomato leaf.",
    AbstainReason.WRONG_CROP: "This does not look like a tomato leaf. This assistant only checks tomato leaves.",
    AbstainReason.LOW_CONFIDENCE: ("The photo does not show a clear match to the problems this assistant knows. "
                                   "Please take another photo, or show the plant to an extension officer."),
    AbstainReason.UNSUPPORTED_LABEL: ("This looks like a tomato problem that this assistant does not cover. "
                                      "Please show the plant to an extension officer."),
}
ESCALATE_ON = {AbstainReason.LOW_CONFIDENCE, AbstainReason.UNSUPPORTED_LABEL}
InputGate = Callable[[str], tuple[str, str]]          # image path -> (result, gate version)


def assumed_tomato_gate(_: str) -> tuple[str, str]:
    """Placeholder until W6 exists: trusts the farmer's crop selection. Recorded in every Finding."""
    return "tomato_leaf", "assumed-dev (W6 input gate not built)"


@dataclass
class StageTimer:
    ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[stage] = (time.perf_counter() - start) * 1000.0


@dataclass(frozen=True)
class PipelineConfig:
    vision: VisionModel
    decision: DecisionConfig
    cards: dict[str, Card]
    sources: dict
    input_gate: InputGate = assumed_tomato_gate
    quality: quality_gate.QualityThresholds = quality_gate.QualityThresholds()
    llama: LlamaConfig | None = None
    cards_reviewed: bool = False


@dataclass(frozen=True)
class PipelineResult:
    finding: Finding
    screen1: Advice
    screen2: Advice | None
    chemical_control_line: str | None
    timings_ms: dict[str, float]

    def to_json(self) -> dict:
        return {"finding": self.finding.model_dump(mode="json"), "screen1": self.screen1.model_dump(mode="json"),
                "screen2": self.screen2.model_dump(mode="json") if self.screen2 else None,
                "chemical_control_line": self.chemical_control_line,
                "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()}}


def _finding(cfg: PipelineConfig, req: Request, *, quality: quality_gate.QualityResult, gate: str = "not_run",
             gate_version: str = "not_run", label: str | None = None, reason: AbstainReason | None = None,
             score: float | None = None) -> Finding:
    return Finding(quality_ok=quality.ok, quality_reason=quality.reason, input_gate_result=gate,
                   input_gate_version=gate_version, crop=req.crop, label=label,
                   unknown_or_unsupported=reason is AbstainReason.UNSUPPORTED_LABEL, model_score=score,
                   abstained=label is None, abstain_reason=reason, model_version=cfg.vision.pte_sha256[:12],
                   threshold_version=cfg.decision.version)


def _screen1(finding: Finding, card: Card | None, lang: str, sources: dict) -> Advice:
    if finding.abstained:
        reason = finding.abstain_reason
        return Advice(lang=lang, card_id=None, knowledge_card_version=None, explanation=ABSTAIN_TEXT[reason],
                      explanation_source="template", escalate=reason in ESCALATE_ON,
                      escalate_reason="the photo could not be matched to a known problem" if reason in ESCALATE_ON else None,
                      contact_source=CONTACT_SOURCE)
    return Advice(lang=card.lang, card_id=card.card_id, knowledge_card_version=card.card_version,
                  actions=card.as_actions(), sources=sources_for(card, sources), explanation=card.template,
                  explanation_source="template", escalate=card.escalate, escalate_reason=card.escalate_reason,
                  contact_source=CONTACT_SOURCE)


@dataclass(frozen=True)
class Screen1Result:
    """Everything Screen 1 shows, plus what Screen 2 needs (the card). The image is not needed after this."""
    finding: Finding
    screen1: Advice
    card: Card | None
    chemical_control_line: str | None
    timings_ms: dict[str, float]


def run(req: Request, cfg: PipelineConfig) -> PipelineResult:
    s1 = run_screen1(req, cfg)
    timings = dict(s1.timings_ms)
    start = time.perf_counter()
    screen2 = run_screen2(s1, req.lang, cfg)
    if screen2 is not None:
        timings["llama_explanation"] = (time.perf_counter() - start) * 1000.0
    return PipelineResult(s1.finding, s1.screen1, screen2, s1.chemical_control_line, timings)


def needs_explanation(s1: Screen1Result) -> bool:
    """Llama is asked only when a problem was named and the card has actions (healthy: nothing to explain;
    21 Nov: the real model refused on the healthy card)."""
    return s1.card is not None and bool(s1.card.actions)


def run_screen2(s1: Screen1Result, lang: Lang, cfg: PipelineConfig) -> Advice | None:
    """Buffered Llama explanation, checked before it can be shown. None when Llama is off or nothing was named."""
    card = s1.card
    if cfg.llama is None or not needs_explanation(s1):
        return None
    prompt = build_prompt(card.crop, card.title.split(" (")[0].lower(), card.symptoms, card.limits, card.cause,
                          reviewed=card.template)
    exp = explain(cfg.llama, prompt, card.template, lang=lang, cause_alternatives=card.cause_alternatives)
    screen2 = s1.screen1.model_copy(update={"explanation": exp.text, "explanation_source": exp.source,
                                            "checker_result": exp.checker_result})
    screen2 = Advice.model_validate(screen2.model_dump())      # re-run validators (checker re-check)
    check_advice_matches(s1.finding, screen2, card)
    return screen2


def run_screen1(req: Request, cfg: PipelineConfig) -> Screen1Result:
    timer = StageTimer()
    with timer("quality_gate"):
        quality = quality_gate.check(req.image_path, cfg.quality)
    finding: Finding
    if not quality.ok:
        finding = _finding(cfg, req, quality=quality, reason=AbstainReason.POOR_QUALITY)
    else:
        with timer("input_gate"):
            gate, gate_version = cfg.input_gate(req.image_path)
        if gate != "tomato_leaf":
            reason = AbstainReason.NOT_A_PLANT if gate == "non_plant" else AbstainReason.WRONG_CROP
            finding = _finding(cfg, req, quality=quality, gate=gate, gate_version=gate_version, reason=reason)
        else:
            with timer("preprocess"):
                batch = cfg.vision.preprocess(req.image_path)
            with timer("vision_inference"):
                logits = cfg.vision.logits(batch)
            with timer("decision"):
                d = decide(logits, cfg.decision, cfg.vision.classes, cfg.vision.pte_sha256)
            finding = _finding(cfg, req, quality=quality, gate=gate, gate_version=gate_version,
                               label=d.label, reason=d.abstain_reason, score=d.score)
    with timer("advice"):
        card = None if finding.abstained else cfg.cards[finding.label]
        screen1 = _screen1(finding, card, req.lang, cfg.sources)
        check_advice_matches(finding, screen1, card)
    timer.ms["screen1_total"] = sum(timer.ms.values())
    chemical_line = CHEMICAL_CONTROL_LINE if card is not None and card.actions else None
    return Screen1Result(finding, screen1, card, chemical_line, dict(timer.ms))


def build_config(pte: str, decision: str, allow_drafts: bool, llama: LlamaConfig | None) -> PipelineConfig:
    vision = load_vision(pte)
    cards, sources = load_cards(allow_drafts=allow_drafts)
    missing = [c for c in vision.classes if c not in cards and c != "tomato_other_problem"]
    if missing:
        raise ValueError(f"no card for model classes {missing}")
    return PipelineConfig(vision=vision, decision=DecisionConfig.load(decision), cards=cards, sources=sources,
                          llama=llama, cards_reviewed=not allow_drafts)


def native_llama_config(root: Path, binary: str) -> LlamaConfig:
    """Pi path: the llama_main binary itself, so a request pays no container start-up."""
    models = root / "artifacts/llama"
    return LlamaConfig(command=(binary,), model_path=str(models / "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"),
                       tokenizer_path=str(models / "tokenizer.model"), timeout_s=60)


def _is_executable(command: str) -> bool:
    """`shutil.which`, or the file's own mode bits: under proot (Android) os.access() can ask about the wrong path."""
    path = Path(command)
    return shutil.which(command) is not None or (path.is_file() and bool(path.stat().st_mode & 0o111))


def default_llama_config(root: Path) -> LlamaConfig:
    """PS1_LLAMA_MAIN (explicit path) > llama_main on PATH > the x86 dev container."""
    explicit = os.environ.get("PS1_LLAMA_MAIN")
    if explicit:
        if not _is_executable(explicit):
            raise FileNotFoundError(f"PS1_LLAMA_MAIN={explicit} is not an executable file")
        return native_llama_config(root, explicit)
    found = shutil.which("llama_main")
    return native_llama_config(root, found) if found else x86_llama_config(root)


def x86_llama_config(root: Path) -> LlamaConfig:
    """Dev path: llama_main inside the G1 container."""
    models = root / "artifacts/llama"
    return LlamaConfig(command=("podman", "run", "--rm", "--network=none", "-v", f"{models}:/models:ro,Z",
                                "ps1-et:0.6.0", "llama_main"),
                       model_path="/models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte",
                       tokenizer_path="/models/tokenizer.model", timeout_s=60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--allow-drafts", action="store_true", help="dev only: use unreviewed cards")
    ap.add_argument("--llama", action="store_true", help="also run the buffered Llama explanation (x86: container)")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = build_config(args.pte, args.decision, args.allow_drafts, default_llama_config(root) if args.llama else None)
    for image in args.image:
        result = run(Request(crop="tomato", lang=args.lang, image_path=image), cfg)
        print(json.dumps({"image": image, **result.to_json()}, indent=1))


if __name__ == "__main__":
    main()
