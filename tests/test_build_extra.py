"""21 Nov: extra field sources go to their own manifest; nothing that duplicates the main manifest gets in."""
import random

import pytest

from ps1.data.build_extra import assign, canonical_for

PH = "8f0f0f0f0f0f0f0f"          # arbitrary 64-bit hashes; equal hashes = near-duplicates
OTHER = "0123456789abcdef"


def row(source_split: str, canonical: str, sha: str, ph: str = OTHER, dh: str = OTHER) -> dict:
    return dict(path=f"x/{sha}.jpg", source="s", source_split=source_split, source_label="l", canonical=canonical,
                sha256=sha, phash=ph, dhash=dh)


@pytest.mark.parametrize("label,expected", [
    ("Bacteiral Spot", "tomato_bacterial_spot"), ("Early Blight Leaf", "tomato_early_blight"),
    ("Healthy", "tomato_healthy"), ("Leaf Miner Flies", "UNSUPPORTED"), ("Yellow Leaf Curl Virus",
    "tomato_yellow_leaf_curl_virus"), ("Tomato Leaf Curl Virus", "tomato_yellow_leaf_curl_virus"),
    ("Healthy Fruit", "EXCLUDED"), ("Class 16 (Uncategorized)", "EXCLUDED"),
    ("Something New", "UNMAPPED")])
def test_folder_labels_map_or_stop(label, expected):
    assert canonical_for(label) == expected


def test_roles_decide_the_split():
    rows = [row("train", "tomato_early_blight", "a", "1" * 16, "1" * 16), row("train", "UNSUPPORTED", "b", "2" * 16, "2" * 16),
            row("test", "tomato_late_blight", "c", "3" * 16, "3" * 16), row("test", "tomato_yellow_leaf_curl_virus", "d", "4" * 16, "4" * 16)]
    assign(rows, main=[])
    assert [r["split"] for r in rows[:2]] == ["x_field_train", "x_ood_train"]
    assert rows[2]["split"] in ("x_dev_known", "x_test_known") and rows[3]["split"] in ("x_dev_ood", "x_test_ood")


def test_copies_of_any_main_image_are_excluded_sealed_included():
    main = [dict(sha256="same", phash="f" * 16, dhash="f" * 16, split="final_external_test"),
            dict(sha256="m2", phash=PH, dhash=PH, split="field_dev")]
    rows = [row("train", "tomato_early_blight", "same", "e" * 16, "e" * 16),
            row("train", "tomato_healthy", "new", PH, PH),
            row("train", "tomato_healthy", "fresh", "1" * 16, "1" * 16)]
    assign(rows, main)
    assert rows[0]["split"] == "excluded" and "final_external_test" in rows[0]["exclude_reason"]
    assert rows[1]["split"] == "excluded" and "near-duplicate" in rows[1]["exclude_reason"]
    assert rows[2]["split"] == "x_field_train"


def test_training_copy_of_a_held_out_test_image_is_dropped_and_test_keeps_it():
    rows = [row("train", "tomato_early_blight", "t1", PH, PH), row("test", "tomato_early_blight", "t2", PH, PH)]
    assign(rows, main=[])
    assert rows[0]["split"] == "excluded" and "held-out" in rows[0]["exclude_reason"]
    assert rows[1]["split"] in ("x_dev_known", "x_test_known")


def test_duplicates_with_conflicting_labels_are_excluded():
    rows = [row("train", "tomato_early_blight", "c1", PH, PH), row("train", "tomato_healthy", "c2", PH, PH)]
    assign(rows, main=[])
    assert all(r["split"] == "excluded" and "conflicting" in r["exclude_reason"] for r in rows)


def test_held_out_source_splits_into_dev_and_test_halves_by_group():
    """22 Nov (Sreyesh): half of the held-out source picks the variant, the other half stays a clean test."""
    rng = random.Random(0)
    hashes = [f"{rng.getrandbits(64):016x}" for _ in range(60)]
    rows = [row("test", "tomato_early_blight", f"s{i:03d}", h, h) for i, h in enumerate(hashes)]
    rows.append(row("test", "tomato_early_blight", "copy_of_s000", hashes[0], hashes[0]))   # near-dup of rows[0]
    assign(rows, main=[])
    splits = [r["split"] for r in rows]
    assert set(splits) == {"x_dev_known", "x_test_known"}
    assert 15 <= splits.count("x_dev_known") <= 45                       # roughly half
    assert rows[0]["split"] == rows[-1]["split"]                         # a duplicate never straddles dev and test
    again = [dict(r) for r in rows]
    assign(again, main=[])
    assert [r["split"] for r in again] == splits                         # deterministic
