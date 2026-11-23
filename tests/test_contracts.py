import pytest
from pydantic import ValidationError

from ps1.contracts import (
    AbstainReason,
    Action,
    Advice,
    Finding,
    QualityReason,
    Request,
    check_advice_matches,
)

BASE = dict(crop="tomato", model_version="m0", threshold_version="t0")


def accepted(**kw):
    return Finding(quality_ok=True, quality_reason=QualityReason.OK, input_gate_result="tomato_leaf",
                   label="tomato_early_blight", model_score=0.9, abstained=False, **BASE, **kw)


def abstained(reason=AbstainReason.LOW_CONFIDENCE, **kw):
    return Finding(quality_ok=True, quality_reason=QualityReason.OK, input_gate_result="tomato_leaf",
                   abstained=True, abstain_reason=reason, **BASE, **kw)


def test_request_defaults_and_rejects_unknown_crop():
    assert Request(crop="tomato", image_path="a.jpg").region == "TS"
    with pytest.raises(ValidationError):
        Request(crop="maize", image_path="a.jpg")


def test_models_are_frozen():
    f = accepted()
    with pytest.raises(ValidationError):
        f.label = "tomato_healthy"


def test_abstained_finding_cannot_carry_label():
    with pytest.raises(ValidationError):
        abstained(label="tomato_early_blight")


def test_abstained_finding_needs_reason():
    with pytest.raises(ValidationError):
        Finding(quality_ok=True, quality_reason=QualityReason.OK, input_gate_result="tomato_leaf",
                abstained=True, **BASE)


def test_accept_requires_both_gates():
    with pytest.raises(ValidationError):
        Finding(quality_ok=False, quality_reason=QualityReason.BLURRY, input_gate_result="tomato_leaf",
                label="tomato_early_blight", abstained=False, **BASE)
    with pytest.raises(ValidationError):
        Finding(quality_ok=True, quality_reason=QualityReason.OK, input_gate_result="non_plant",
                label="tomato_early_blight", abstained=False, **BASE)


def test_severity_always_unavailable():
    with pytest.raises(ValidationError):
        accepted(severity_available=True)


def test_no_card_means_no_actions():
    with pytest.raises(ValidationError):
        Advice(lang="en", card_id=None, knowledge_card_version=None,
               actions=(Action(action_id="a1", text="x", source_id="s"),), escalate=False)


def test_llama_text_requires_checker_pass():
    with pytest.raises(ValidationError):
        Advice(lang="en", card_id="c1", knowledge_card_version="1", explanation="Looks like early blight.",
               explanation_source="llama", checker_result="fail", escalate=False)
    ok = Advice(lang="en", card_id="c1", knowledge_card_version="1",
                explanation="Looks like early blight. A photo cannot confirm it.",
                explanation_source="llama", checker_result="pass", escalate=False)
    assert ok.explanation_source == "llama"


@pytest.mark.parametrize("text", ["hi", "Spray mancozeb now.", "This is definitely early blight."])
def test_caller_claimed_pass_cannot_smuggle_unchecked_text(text):
    with pytest.raises(ValidationError):
        Advice(lang="en", card_id="c1", knowledge_card_version="1", explanation=text,
               explanation_source="llama", checker_result="pass", escalate=False)


def test_escalation_needs_reason():
    with pytest.raises(ValidationError):
        Advice(lang="en", card_id=None, knowledge_card_version=None, escalate=True)


class FakeCard:
    def __init__(self, label="tomato_early_blight", crop="tomato", card_id="c1", version="1", lang="en", ids=("a1",)):
        self.label, self.crop, self.card_id, self.card_version, self.lang = label, crop, card_id, version, lang
        self._actions = tuple(Action(action_id=i, text="x.", source_id="s") for i in ids)

    def as_actions(self):
        return self._actions


def advice_for(card):
    return Advice(lang="en", card_id=card.card_id, knowledge_card_version=card.card_version,
                  actions=card.as_actions(), escalate=False)


def test_abstained_finding_cannot_get_card():
    card = FakeCard()
    with pytest.raises(ValueError):
        check_advice_matches(abstained(), advice_for(card), card)
    check_advice_matches(abstained(), Advice(lang="en", card_id=None, knowledge_card_version=None, escalate=False), None)


def test_card_must_match_finding_label():
    card = FakeCard(label="tomato_late_blight")
    with pytest.raises(ValueError, match="finding is"):
        check_advice_matches(accepted(), advice_for(card), card)


def test_actions_and_version_must_equal_the_card():
    card = FakeCard(ids=("a1", "a2"))
    check_advice_matches(accepted(), advice_for(card), card)
    with pytest.raises(ValueError):
        check_advice_matches(accepted(), advice_for(FakeCard(ids=("a1",))), card)
    with pytest.raises(ValueError):
        check_advice_matches(accepted(), advice_for(FakeCard(version="2", ids=("a1", "a2"))), card)


def test_quality_flag_must_agree_with_reason():
    with pytest.raises(ValidationError):
        Finding(quality_ok=True, quality_reason=QualityReason.BLURRY, input_gate_result="tomato_leaf",
                abstained=True, abstain_reason=AbstainReason.POOR_QUALITY, **BASE)


@pytest.mark.parametrize("quality_ok,reason,gate,expected", [
    (False, QualityReason.BLURRY, "not_run", AbstainReason.POOR_QUALITY),
    (True, QualityReason.OK, "non_plant", AbstainReason.NOT_A_PLANT),
    (True, QualityReason.OK, "other_plant", AbstainReason.WRONG_CROP),
])
def test_failed_gate_forces_its_abstain_reason(quality_ok, reason, gate, expected):
    Finding(quality_ok=quality_ok, quality_reason=reason, input_gate_result=gate,
            abstained=True, abstain_reason=expected, **BASE)
    with pytest.raises(ValidationError):
        Finding(quality_ok=quality_ok, quality_reason=reason, input_gate_result=gate,
                abstained=True, abstain_reason=AbstainReason.LOW_CONFIDENCE, **BASE)


def test_advice_validator_reruns_llama_guards_not_only_lexical_rules():
    import pytest as _pt
    from ps1.contracts import Advice
    with _pt.raises(ValueError):
        Advice(lang="en", card_id=None, knowledge_card_version=None, escalate=False,
               explanation="You may be seeing early blight on the older leaves.",   # no photo warning
               explanation_source="llama", checker_result="pass")
