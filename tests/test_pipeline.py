import sys
import textwrap

import cv2
import numpy as np
import pytest

from ps1.cards import load_cards
from ps1.contracts import AbstainReason, Request
from ps1.decision import DecisionConfig, decide
from ps1.llm.explain import LlamaConfig
from ps1.pipeline import PipelineConfig, run
from ps1.vision.runtime import VisionModel

CLASSES = ("tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_spot", "tomato_other_problem")
SHA = "f" * 64


class FakeModule:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)

    def forward(self, inputs):
        import torch
        assert tuple(inputs[0].shape) == (1, 3, 224, 224)
        return [torch.from_numpy(self.logits[None, :])]


def logits_for(label: str, margin: float = 8.0) -> list[float]:
    z = [0.0] * len(CLASSES)
    z[CLASSES.index(label)] = margin
    return z


def cfg(logits, gate=None, llama=None) -> PipelineConfig:
    cards, sources = load_cards(allow_drafts=True)
    vision = VisionModel(CLASSES, SHA, "t2", "fp32", FakeModule(logits))
    decision = DecisionConfig("d-test", SHA, CLASSES, temperature=1.0, threshold=0.7, source="test")
    kwargs = {"input_gate": gate} if gate else {}
    return PipelineConfig(vision=vision, decision=decision, cards=cards, sources=sources, llama=llama, **kwargs)


@pytest.fixture
def leaf(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "leaf.png"
    cv2.imwrite(str(path), rng.integers(60, 200, size=(480, 640, 3), dtype=np.uint8))
    return str(path)


@pytest.fixture
def blurry(tmp_path):
    path = tmp_path / "blur.png"
    img = np.random.default_rng(1).integers(60, 200, size=(480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(path), cv2.GaussianBlur(img, (0, 0), sigmaX=8))
    return str(path)


def req(path):
    return Request(crop="tomato", image_path=path)


# --- decision engine ---------------------------------------------------------------------------
def test_decide_accept_low_confidence_and_other():
    c = DecisionConfig("d", SHA, CLASSES, 1.0, 0.7, "test")
    assert decide(np.array(logits_for("tomato_late_blight")), c, CLASSES, SHA).label == "tomato_late_blight"
    flat = decide(np.zeros(len(CLASSES)), c, CLASSES, SHA)
    assert flat.label is None and flat.abstain_reason is AbstainReason.LOW_CONFIDENCE
    other = decide(np.array(logits_for("tomato_other_problem")), c, CLASSES, SHA)
    assert other.label is None and other.abstain_reason is AbstainReason.UNSUPPORTED_LABEL


def test_decision_config_bound_to_its_model():
    c = DecisionConfig("d", SHA, CLASSES, 1.0, 0.7, "test")
    with pytest.raises(ValueError):
        decide(np.zeros(len(CLASSES)), c, CLASSES, "0" * 64)
    with pytest.raises(ValueError):
        decide(np.zeros(len(CLASSES)), c, tuple(reversed(CLASSES)), SHA)


# --- pipeline branches -------------------------------------------------------------------------
def test_blurry_photo_abstains_before_the_model(blurry):
    r = run(req(blurry), cfg(logits_for("tomato_early_blight")))
    assert r.finding.abstain_reason is AbstainReason.POOR_QUALITY
    assert r.screen1.card_id is None and not r.screen1.actions
    assert "vision_inference" not in r.timings_ms


def test_non_plant_gate_blocks_diagnosis(leaf):
    r = run(req(leaf), cfg(logits_for("tomato_early_blight"), gate=lambda _: ("non_plant", "test-gate")))
    assert r.finding.abstain_reason is AbstainReason.NOT_A_PLANT and r.screen1.card_id is None


def test_accepted_disease_gets_its_card_kcc_line_and_timings(leaf):
    r = run(req(leaf), cfg(logits_for("tomato_early_blight")))
    assert r.finding.label == "tomato_early_blight" and not r.finding.abstained
    assert r.finding.input_gate_version.startswith("assumed-dev")
    assert r.screen1.card_id == "tomato_early_blight" and r.screen1.actions
    assert all(a.source_id for a in r.screen1.actions)
    assert r.chemical_control_line is not None
    for stage in ("quality_gate", "input_gate", "preprocess", "vision_inference", "decision", "advice", "screen1_total"):
        assert stage in r.timings_ms
    assert r.screen2 is None


def test_healthy_has_no_actions_and_no_chemical_line(leaf):
    r = run(req(leaf), cfg(logits_for("tomato_healthy")))
    assert r.screen1.card_id == "tomato_healthy" and not r.screen1.actions
    assert r.chemical_control_line is None


def test_other_problem_is_unsupported_and_escalated(leaf):
    r = run(req(leaf), cfg(logits_for("tomato_other_problem")))
    assert r.finding.abstain_reason is AbstainReason.UNSUPPORTED_LABEL and r.finding.unknown_or_unsupported
    assert r.screen1.escalate and r.screen1.card_id is None


# --- Screen 2 through a fake Llama runner ------------------------------------------------------
def fake_llama(tmp_path, generated: str) -> LlamaConfig:
    script = tmp_path / "fake_llama.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        prompt = [a for a in sys.argv if a.startswith("--prompt=")][0][len("--prompt="):]
        print(prompt + {generated!r} + "<|eot_id|>")
        print('PyTorchObserver {{"prompt_tokens":10,"generated_tokens":5}}')
    """))
    return LlamaConfig(command=(sys.executable, str(script)), model_path="m", tokenizer_path="t", timeout_s=10)


def test_screen2_reveals_checked_llama_text(leaf, tmp_path):
    text = "The ringed brown spots on older leaves look like early blight. A photo cannot tell for certain."
    r = run(req(leaf), cfg(logits_for("tomato_early_blight"), llama=fake_llama(tmp_path, text)))
    assert r.screen2.explanation_source == "llama" and r.screen2.explanation == text
    assert r.screen2.actions == r.screen1.actions                     # Llama never changes actions
    assert "llama_explanation" in r.timings_ms


def test_screen2_unsafe_llama_text_falls_back_to_template(leaf, tmp_path):
    r = run(req(leaf), cfg(logits_for("tomato_early_blight"), llama=fake_llama(tmp_path, "Spray mancozeb weekly.")))
    assert r.screen2.explanation_source == "template" and r.screen2.checker_result == "fail"
    assert r.screen2.explanation == r.screen1.explanation


# --- integration: the real exported model (skipped if the artifact is absent) -------------------
REAL_PTE = "runs/v4_f0/export/tomato_effnet_b0_t2_fp32_xnnpack.pte"


@pytest.mark.skipif(not __import__("pathlib").Path(REAL_PTE).exists(), reason="v4 FP32 export not present")
def test_real_pte_runs_end_to_end_after_loader_returns(leaf):
    import gc

    from ps1.vision.runtime import load_vision
    vision = load_vision(REAL_PTE)
    gc.collect()                                     # the loader's local buffer is gone; model must still run
    out = vision.logits(vision.preprocess(leaf))
    assert out.shape == (len(vision.classes),) and np.isfinite(out).all()


def test_healthy_never_calls_llama(leaf, tmp_path):
    r = run(req(leaf), cfg(logits_for("tomato_healthy"), llama=fake_llama(tmp_path, "anything")))
    assert r.screen2 is None and "llama_explanation" not in r.timings_ms


def test_referral_names_the_extension_officer_not_a_phone_line():
    # The advice must work with no network: it points to the local extension officer, never to a call-centre number.
    from ps1.cards import CHEMICAL_CONTROL_LINE, CONTACT_SOURCE
    for text in (CHEMICAL_CONTROL_LINE, CONTACT_SOURCE):
        assert "extension officer" in text
        assert "Kisan" not in text and "1800" not in text
