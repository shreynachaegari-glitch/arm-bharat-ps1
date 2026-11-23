import shutil

import pytest

from ps1.cards import CARDS_DIR, load_cards
from ps1.data.build_splits import SUPPORTED
from ps1.llm.checker import CHEMICAL_RE, DIGIT
from ps1.vision.dataset import TAXONOMIES

T2_CLASSES = {TAXONOMIES["t2"].get(c, c) for c in SUPPORTED}


def test_every_t2_class_has_exactly_one_card():
    cards, _ = load_cards(allow_drafts=True)
    assert set(cards) == T2_CLASSES


def test_unreviewed_cards_are_refused_by_default():
    with pytest.raises(ValueError, match="unreviewed card cannot be shipped"):
        load_cards()


def test_card_actions_are_non_chemical_numberless_and_sourced():
    cards, sources = load_cards(allow_drafts=True)
    for card in cards.values():
        for a in card.as_actions():
            assert not CHEMICAL_RE.search(a.text), (card.card_id, a.text)
            assert not DIGIT.search(a.text), (card.card_id, a.text)
            assert a.source_id in sources


def test_unknown_source_rejected(tmp_path):
    shutil.copytree(CARDS_DIR / "en", tmp_path / "en")
    path = tmp_path / "en" / "tomato_early_blight.yaml"
    path.write_text(path.read_text().replace("source: ucipm_early_blight}", "source: made_up}", 1))
    with pytest.raises(ValueError, match="unknown source"):
        load_cards(root=tmp_path, allow_drafts=True)


def test_template_must_pass_checker(tmp_path):
    shutil.copytree(CARDS_DIR / "en", tmp_path / "en")
    path = tmp_path / "en" / "tomato_healthy.yaml"
    path.write_text(path.read_text().replace("keep checking the plants.", "spray mancozeb."))
    with pytest.raises(ValueError, match="template fails the checker"):
        load_cards(root=tmp_path, allow_drafts=True)


def test_leaf_spot_card_declares_its_two_possible_causes():
    from ps1.cards import load_cards
    cards, _ = load_cards(allow_drafts=True)
    groups = cards["tomato_leaf_spot"].cause_alternatives
    assert len(groups) == 2 and all(g and all(t == t.lower() for t in g) for g in groups)
