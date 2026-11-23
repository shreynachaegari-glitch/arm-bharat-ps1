"""Request / Finding / Advice contracts (PS1 Master Plan v4, section 4).

All models are frozen. The invariants from v4 are enforced here so that no
pipeline stage, UI or test can build an object that breaks them:

* an abstained finding never carries a disease label;
* an absent/unsupported card yields no disease-specific actions;
* LLM text is only shown after the checker passed it (explanation_source="llama");
* severity is unavailable in Phase 1.
"""
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ps1.llm.checker import check_llama_text, keeps_cause_alternatives

CONTRACT_VERSION = "1.0.0"

Crop = Literal["tomato"]                    # maize only after the tomato slice is stable (v4 cut order)
Lang = Literal["en", "te", "hi"]
InputGateResult = Literal["tomato_leaf", "other_plant", "non_plant", "not_run"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class QualityReason(str, Enum):
    OK = "ok"
    BLURRY = "blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    UNREADABLE = "unreadable"


class AbstainReason(str, Enum):
    POOR_QUALITY = "poor_quality"
    NOT_A_PLANT = "not_a_plant"
    WRONG_CROP = "wrong_crop"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_LABEL = "unsupported_label"


class Request(Frozen):
    crop: Crop
    region: Literal["TS"] = "TS"
    district: str | None = None             # optional selector, no GPS (v4 2.13)
    lang: Lang = "en"
    image_path: str


class Finding(Frozen):
    quality_ok: bool
    quality_reason: QualityReason
    input_gate_result: InputGateResult
    input_gate_version: str = "not_run"     # which gate produced the result (e.g. "assumed-dev" until W6)
    crop: Crop
    label: str | None = None                # canonical class id from taxonomy_map.csv
    unknown_or_unsupported: bool = False
    model_score: float | None = Field(default=None, ge=0.0, le=1.0)  # internal only, never shown as a probability
    abstained: bool
    abstain_reason: AbstainReason | None = None
    model_version: str
    threshold_version: str
    severity_available: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "Finding":
        if self.quality_ok != (self.quality_reason is QualityReason.OK):
            raise ValueError("quality_ok must be true exactly when quality_reason is OK")
        # Each failed gate forces abstention with its own reason, so no later stage can diagnose past it.
        gate_reason = (AbstainReason.POOR_QUALITY if not self.quality_ok else
                       AbstainReason.NOT_A_PLANT if self.input_gate_result == "non_plant" else
                       AbstainReason.WRONG_CROP if self.input_gate_result == "other_plant" else None)
        if gate_reason is not None and not (self.abstained and self.abstain_reason is gate_reason):
            raise ValueError(f"a failed gate requires abstained=True with abstain_reason={gate_reason.value}")
        if self.abstained:
            if self.label is not None:
                raise ValueError("abstained finding must not carry a label")
            if self.abstain_reason is None:
                raise ValueError("abstained finding needs an abstain_reason")
        else:
            if self.label is None:
                raise ValueError("accepted finding needs a label")
            if self.abstain_reason is not None or self.unknown_or_unsupported:
                raise ValueError("accepted finding cannot have an abstain reason or be unsupported")
            if not self.quality_ok or self.input_gate_result != "tomato_leaf":
                raise ValueError("finding can only be accepted after the quality and input gates pass")
        return self


class Action(Frozen):
    action_id: str
    text: str
    source_id: str                          # key into the card's sources: every action shows its evidence


class Source(Frozen):
    title: str
    url: str
    accessed: str                           # ISO date the source was read


class Advice(Frozen):
    lang: Lang
    card_id: str | None
    knowledge_card_version: str | None
    actions: tuple[Action, ...] = ()        # selected by deterministic code from the card, never by the LLM
    sources: tuple[Source, ...] = ()
    explanation: str = ""
    explanation_source: Literal["llama", "template", "none"] = "none"
    checker_result: Literal["pass", "fail", "timeout", "error", "not_run"] = "not_run"
    escalate: bool
    escalate_reason: str | None = None
    contact_source: str | None = None

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(a.action_id for a in self.actions)

    @model_validator(mode="after")
    def _consistent(self) -> "Advice":
        if self.card_id is None and self.actions:
            raise ValueError("no card means no disease-specific actions")
        if self.card_id is not None and self.knowledge_card_version is None:
            raise ValueError("card advice must record the card version")
        if self.explanation_source == "llama":
            # Never trust a caller's "pass": re-run the checker on the exact text that would be shown.
            verdict = check_llama_text(self.explanation, lang=self.lang)
            if self.checker_result != "pass" or not verdict.passed:
                raise ValueError(f"Llama text fails the checker {verdict.version}: {verdict.reasons}")
        if self.escalate and not self.escalate_reason:
            raise ValueError("escalation needs a reason")
        return self


class CardLike(Protocol):
    card_id: str
    card_version: str
    crop: str
    label: str
    lang: str

    def as_actions(self) -> tuple[Action, ...]: ...


def check_advice_matches(finding: Finding, advice: Advice, card: CardLike | None) -> None:
    """Cross-object invariants between a finding, the card chosen for it and the advice built from it."""
    if finding.abstained or card is None:
        if advice.card_id is not None or advice.actions:
            raise ValueError("an abstained finding (or no card) must not get a disease card or actions")
        return
    if (card.label, card.crop) != (finding.label, finding.crop):
        raise ValueError(f"card {card.card_id} is for {card.crop}/{card.label}, finding is {finding.crop}/{finding.label}")
    if (advice.card_id, advice.knowledge_card_version, advice.lang) != (card.card_id, card.card_version, card.lang):
        raise ValueError("advice card id / version / language differ from the selected card")
    if advice.actions != card.as_actions():
        raise ValueError(f"actions {advice.action_ids} differ from the card's actions")
    if advice.explanation_source == "llama" and not keeps_cause_alternatives(
            advice.explanation, getattr(card, "cause_alternatives", ())):
        raise ValueError("Llama text names only one of the card's possible causes")
