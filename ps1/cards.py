"""Knowledge cards: versioned YAML source of truth (v4 section 4; addenda #2).

Cards carry reviewed, cited, non-chemical actions only. The one chemical-control line is a
fixed constant rendered by code (addenda #2), so no card or model can vary it.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ps1.contracts import Action, Source
from ps1.llm.checker import check_explanation

ReviewStatus = Literal["draft_unreviewed", "reviewed"]
CARDS_DIR = Path(__file__).resolve().parent.parent / "cards"
CHEMICAL_CONTROL_LINE = (
    "If chemical control is needed, use only a product registered for this crop and disease; "
    "ask your local agriculture extension officer for the product and dose."
)
CONTACT_SOURCE = "Your local agriculture extension officer"


class CardAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    text: str
    source: str


class Card(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    card_id: str
    card_version: str
    review_status: ReviewStatus
    crop: str
    region: str                     # "general" unless the evidence is specific to Telangana ("TS")
    applicable_districts: tuple[str, ...] = ()   # empty = not district-specific
    label: str
    lang: str
    title: str
    cause: str
    symptoms: str
    limits: str
    actions: tuple[CardAction, ...]
    escalate: bool
    escalate_reason: str | None = None
    template: str
    cause_alternatives: tuple[tuple[str, ...], ...] = ()   # lowercase stems per possible cause; Llama names all or none
    reviewer: str | None = None
    reviewed_on: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Card":
        if self.escalate and not self.escalate_reason:
            raise ValueError(f"{self.card_id}: escalate needs escalate_reason")
        if self.review_status == "reviewed" and not (self.reviewer and self.reviewed_on):
            raise ValueError(f"{self.card_id}: reviewed card needs reviewer and reviewed_on")
        if any(not group or any(t != t.lower() or not t for t in group) for group in self.cause_alternatives):
            raise ValueError(f"{self.card_id}: cause_alternatives must be non-empty groups of lowercase stems")
        verdict = check_explanation(self.template, lang="en")
        if self.lang == "en" and not verdict.passed:
            raise ValueError(f"{self.card_id}: template fails the checker: {verdict.reasons}")
        return self

    def as_actions(self) -> tuple[Action, ...]:
        return tuple(Action(action_id=a.id, text=a.text, source_id=a.source) for a in self.actions)


def load_cards(lang: str = "en", root: Path = CARDS_DIR,
               allow_drafts: bool = False) -> tuple[dict[str, Card], dict[str, Source]]:
    """Cards keyed by label, plus the shared sources; raises on any broken reference.

    Fails closed: an unreviewed card is refused unless allow_drafts=True, which only dev runs and
    tests may pass (v4: every displayed action comes from a reviewed card).
    """
    folder = root / lang
    raw_sources = yaml.safe_load((folder / "_sources.yaml").read_text())
    sources = {k: Source(**v) for k, v in raw_sources.items()}
    cards: dict[str, Card] = {}
    seen_action_ids: set[str] = set()
    for path in sorted(folder.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        card = Card(**yaml.safe_load(path.read_text()))
        if card.review_status != "reviewed" and not allow_drafts:
            raise ValueError(f"unreviewed card cannot be shipped: {card.card_id} ({path.name})")
        if card.label in cards:
            raise ValueError(f"duplicate card for {card.label}")
        for a in card.actions:
            if a.source not in sources:
                raise ValueError(f"{card.card_id}/{a.id}: unknown source {a.source}")
            if a.id in seen_action_ids:
                raise ValueError(f"duplicate action id {a.id}")
            seen_action_ids.add(a.id)
        cards[card.label] = card
    return cards, sources


def sources_for(card: Card, sources: dict[str, Source]) -> tuple[Source, ...]:
    used = dict.fromkeys(a.source for a in card.actions)
    return tuple(sources[s] for s in used)
