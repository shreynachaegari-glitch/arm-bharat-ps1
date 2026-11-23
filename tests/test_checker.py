import pytest

from ps1.llm.checker import check_explanation

GOOD = ("The dark brown spots with rings on the older leaves look like early blight. "
        "A photo cannot confirm the disease, and other problems can look similar.")


def test_clean_explanation_passes():
    r = check_explanation(GOOD)
    assert r.passed, r.reasons


@pytest.mark.parametrize("text,reason_prefix", [
    ("Spray mancozeb on the plant.", "chemical/dose word"),
    ("Use a copper based product weekly.", "chemical/dose word"),
    ("Mix two grams per litre.", "chemical/dose word"),
    ("Apply 2 times a week.", "contains digits"),
    ("This is definitely early blight.", "certainty word"),
    ("The tomatoes are safe to eat.", "harvest claim"),
    ("This looks like early blight.<|eot_id|>", "special token leaked"),
    ("ఇది ఎర్లీ బ్లైట్ లాగా ఉంది.", "non-English script"),
    ("", "empty"),
])
def test_unsafe_or_malformed_output_blocked(text, reason_prefix):
    r = check_explanation(text)
    assert not r.passed
    assert any(reason.startswith(reason_prefix) for reason in r.reasons), r.reasons


def test_too_many_sentences_blocked():
    assert not check_explanation("One. Two. Three. Four.").passed


def test_too_long_blocked():
    assert not check_explanation("word " * 120).passed


def test_generated_non_english_disabled_in_phase1():
    assert not check_explanation(GOOD, lang="te").passed


def test_no_false_positive_on_words_containing_units():
    assert check_explanation("The background and small spots look like early blight.").passed


@pytest.mark.parametrize("hedged", [
    "The spots look like early blight, but this cannot be confirmed from a photo.",
    "Early blight is likely, although it cannot be diagnosed from a photo alone.",
])
def test_hedged_negations_pass(hedged):
    r = check_explanation(hedged)
    assert r.passed, r.reasons


@pytest.mark.parametrize("certain", [
    "This is confirmed to be early blight.",
    "The plant is diagnosed with early blight.",
    "The diagnosis is early blight.",
])
def test_affirmed_certainty_still_blocked(certain):
    assert not check_explanation(certain).passed


def test_truncated_output_blocked():
    r = check_explanation("The spots look like early blight because the")
    assert not r.passed and "incomplete sentence" in r.reasons


# --- c3 (21 Nov): a NEGATED certainty phrase is a hedge, not a claim (real model: "A photo can't tell for sure")
@pytest.mark.parametrize("text", [
    "A photo can't tell for sure, and other spots can look the same.",
    "A photo cannot tell for certain which one it is.",
    "You cannot be sure from one photo.",
])
def test_negated_certainty_is_a_hedge(text):
    assert check_explanation(text).passed, check_explanation(text).reasons


@pytest.mark.parametrize("text", [
    "This is early blight for sure.",
    "It is definitely late blight, so you can't tell for sure about the rest.",   # affirmative part still caught
    "We can say for sure it is blight.",
])
def test_affirmative_certainty_still_fails(text):
    assert not check_explanation(text).passed


# --- review 21 Nov: affirmative claims hidden inside a negation must still fail -----------------------------------
@pytest.mark.parametrize("text", [
    "It cannot be for sure anything but blight.",
    "The spots cannot be certainly anything but blight.",
    "This can not be surely a fungus problem, it is a bacterial problem.",
    "A photo cannot tell for sure anything other than blight.",
])
def test_negation_wrapped_certainty_claims_fail(text):
    assert not check_explanation(text).passed


@pytest.mark.parametrize("text,ok", [
    ("I cannot provide advice on treating a plant disease. Can I help you with something else?", False),
    ("You may be seeing early blight on the older leaves.", False),
    ("The photo does not show a spot, it is blight.", False),          # 'photo does not' is not the warning
    ("You may be seeing early blight. A photo cannot confirm it.", True),
    ("A photo can't tell which one it is, and other problems can look similar.", True),
])
def test_llama_text_guards(text, ok):
    from ps1.llm.checker import check_llama_text
    assert check_llama_text(text).passed is ok
