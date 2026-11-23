"""Checker for buffered Llama output (v4 P0-2, 2.11; addenda #3).

This is a lexical guardrail, not semantic verification: it blocks digits,
chemical/dose words, certainty claims, harvest/food-safety claims, wrong
script, leftover special tokens and over-long text. Anything it rejects is
replaced by the reviewed template. Actions never come from the LLM anyway.
"""
import re
from dataclasses import dataclass

CHECKER_VERSION = "c4"   # c2: affirmative-only certainty; c3: negated hedges; c4: narrower hedge + Llama guards
MAX_CHARS = 450
MAX_SENTENCES = 3

# Matched case-insensitively at a word start (so plurals match). Generic chemical words plus
# common active ingredients sold in India, so a paraphrase like "a copper spray" is still caught.
CHEMICAL_LEXICON = (
    "fungicide", "pesticide", "insecticide", "herbicide", "bactericide", "chemical", "spray",
    "dose", "dosage", "litre", "liter", "gram", "tank",
    "mancozeb", "chlorothalonil", "copper", "carbendazim", "azoxystrobin", "metalaxyl",
    "propiconazole", "hexaconazole", "tebuconazole", "difenoconazole", "imidacloprid",
    "thiamethoxam", "acephate", "chlorpyrifos", "cypermethrin", "streptomycin", "sulphur", "sulfur",
    "neem oil", "bordeaux",
)
UNIT_WORDS = ("ml", "kg", "ppm", "g", "l", "ha")   # matched as whole words only
CHEMICAL_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, CHEMICAL_LEXICON)) + r")", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(?:" + "|".join(UNIT_WORDS) + r")\b", re.IGNORECASE)
# Affirmative certainty only: bare "confirmed"/"diagnosed" would also block the hedge the cards ask
# for ("this cannot be confirmed from a photo").
CERTAINTY_WORDS = (
    "definitely", "certainly", "100%", "guarantee", "surely", "without doubt", "no doubt", "for sure",
    "is sure to", "confirm that", "is confirmed", "confirmed to be", "confirmed as", "diagnosed with",
    "diagnosed as", "diagnosis is", "it is clearly",
)
# c3: "a photo can't tell for sure" is a hedge; it is removed before the certainty scan (affirmative uses remain).
NEGATED_HEDGE = re.compile(
    r"\b(cannot|can't|can’t|can not|not|never)\s+(tell|say|know)\s+(for sure|for certain|certainly|surely)\b"
    r"(?!\s+(anything|other|otherwise|but|except))",   # review 21 Nov: "cannot tell for sure anything but X" is a claim
    re.IGNORECASE)
HARVEST_CLAIMS = ("safe to eat", "harvest is safe", "safe to harvest", "safe for consumption")

DIGIT = re.compile(r"\d")
SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
# Phase 1 English only: Latin letters, spaces and basic punctuation.
ALLOWED_ENGLISH = re.compile(r"^[A-Za-z\s.,;:'\"()\-!?]+$")


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reasons: tuple[str, ...]
    version: str = CHECKER_VERSION


def check_explanation(text: str, lang: str = "en") -> CheckResult:
    reasons: list[str] = []
    stripped = text.strip()
    lower = f" {stripped.lower()} "

    if not stripped:
        reasons.append("empty")
    if lang != "en":
        reasons.append("generated text is English-only in Phase 1")
    if SPECIAL_TOKEN.search(stripped):
        reasons.append("special token leaked")
    if DIGIT.search(stripped):
        reasons.append("contains digits")
    if stripped and not ALLOWED_ENGLISH.match(SPECIAL_TOKEN.sub("", stripped)):
        reasons.append("non-English script or symbols")
    if len(stripped) > MAX_CHARS:
        reasons.append("too long")
    if len(SENTENCE_END.findall(stripped)) > MAX_SENTENCES:
        reasons.append("too many sentences")
    if stripped and not stripped.endswith((".", "!", "?")):
        reasons.append("incomplete sentence")      # e.g. cut off at the token limit
    reasons += [f"chemical/dose word: {m.group(0).lower()}" for m in CHEMICAL_RE.finditer(stripped)]
    reasons += [f"chemical/dose word: {m.group(0).lower()}" for m in UNIT_RE.finditer(stripped)]
    unhedged = NEGATED_HEDGE.sub(" ", lower)
    reasons += [f"certainty word: {w}" for w in CERTAINTY_WORDS if w in unhedged]
    reasons += [f"harvest claim: {w}" for w in HARVEST_CLAIMS if w in lower]

    return CheckResult(passed=not reasons, reasons=tuple(reasons))


# --- c4: extra rules for generated (Llama) text only; card templates are reviewed and use check_explanation ------
REFUSAL = re.compile(r"\b(i cannot|i can't|i can not|i'm not able|i am not able|can i help|as an ai)\b", re.IGNORECASE)
PHOTO_LIMIT = re.compile(
    r"\bphoto\b[^.]*\b(cannot|can't|can’t|can not|is not able to|isn't able to)\s+(confirm|tell|say|show|be sure)\b",
    re.IGNORECASE)


def check_llama_text(text: str, lang: str = "en") -> CheckResult:
    """check_explanation plus: no refusal, and the 'a photo cannot confirm' warning must survive (English)."""
    base = check_explanation(text, lang=lang)
    reasons = list(base.reasons)
    if REFUSAL.search(text):
        reasons.append("model refused")
    if lang == "en" and not PHOTO_LIMIT.search(text):
        reasons.append("drops the 'a photo cannot confirm' warning")
    return CheckResult(passed=not reasons, reasons=tuple(reasons))


def keeps_cause_alternatives(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    """If the text names any one possible cause, it must name all of them (e.g. fungus AND bacteria).

    `groups` comes from the card: each group is a set of lowercase stems for one cause. Text naming no cause
    at all is fine (the reviewed card already explains it); naming only one turns "either" into a diagnosis.
    """
    lowered = text.lower()
    named = [any(stem in lowered for stem in group) for group in groups]
    return not any(named) or all(named)
