from collections import defaultdict

import pytest

from ps1.data.build_splits import SEALED, assign, group_duplicates


def test_exact_and_near_duplicates_grouped():
    sha = ["a", "a", "b", "c"]
    phash = ["0000000000000000", "ffffffffffffffff", "0000000000000003", "ffffffffffffffff"]
    g = group_duplicates(sha, phash)
    assert g[0] == g[1]            # same sha256
    assert g[0] == g[2]            # pHash 2 bits away from row 0
    assert g[3] == g[1]            # same pHash as row 1, which is grouped with row 0


def test_dhash_must_confirm_phash_match():
    sha = ["a", "b"]
    phash = ["0000000000000000", "0000000000000001"]          # 1 bit apart: pHash says duplicate
    assert len(set(group_duplicates(sha, phash, ["0000000000000000", "ffffffffffffffff"]))) == 2
    assert len(set(group_duplicates(sha, phash, ["0000000000000000", "0000000000000001"]))) == 1


def row(source, source_split, canonical, group, phash="0f"):
    return dict(source=source, source_split=source_split, canonical=canonical, group=group, phash=phash)


def test_pools_route_to_the_right_splits():
    rows = [
        *[row("plantdoc", "test" if i % 2 else "train", "tomato_early_blight", group=i) for i in range(40)],
        *[row("plantvillage", "all", "tomato_late_blight", group=100 + i) for i in range(20)],
        *[row("plantvillage", "all", "tomato_leaf_mold", group=200 + i) for i in range(6)],     # not shipped -> OOD
        *[row("plantdoc", "train", "UNSUPPORTED", group=300 + i) for i in range(6)],
        *[row("plantvillage", "all", "OTHER_CROP", group=400 + i) for i in range(20)],
    ]
    assign(rows)
    pd = [r["split"] for r in rows[:40]]
    assert set(pd) == {"field_dev", SEALED} and pd.count(SEALED) == 20       # PD pooled, 50/50
    assert {r["split"] for r in rows[40:60]} <= {"pv_train", "pv_val", "pv_calib"}
    assert {r["split"] for r in rows[60:66]} <= {"ood_train", "ood_test"}      # lab OOD: outlier exposure
    assert {r["split"] for r in rows[66:72]} <= {"ood_dev", "ood_test"}        # field OOD stays dev/test
    assert {r["split"] for r in rows[60:72]} >= {"ood_train", "ood_dev"}
    assert {r["split"] for r in rows[72:]} <= {"other_train", "other_dev", "other_test"}


def test_sealed_split_keeps_duplicate_group():
    rows = [row("plantdoc", "test", "tomato_early_blight", group=i) for i in range(10)]
    rows += [row("plantvillage", "all", "tomato_early_blight", group=i) for i in range(10)]  # PV dup of each PD row
    rows += [row("plantvillage", "all", "tomato_early_blight", group=50 + i) for i in range(20)]
    assign(rows)
    for pd_row, pv_row in zip(rows[:10], rows[10:20]):
        assert pv_row["split"] == "excluded"
        if pd_row["split"] == SEALED:
            assert "sealed" in pv_row["exclude_reason"]


def test_groups_never_straddle_splits():
    rows = [row("plantvillage", "all", "tomato_healthy", group=i // 3) for i in range(60)]
    rows.append(row("plantdoc", "train", "tomato_healthy", group=0))   # duplicate of a PV group
    assign(rows)
    splits_per_group = defaultdict(set)
    for r in rows:
        if r["split"] != "excluded":
            splits_per_group[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())


def test_unreadable_images_excluded():
    rows = [row("plantvillage", "all", "tomato_healthy", group=0, phash="")]
    assign(rows)
    assert rows[0]["split"] == "excluded"


def test_loader_refuses_every_sealed_split(monkeypatch):
    from ps1.vision.dataset import select
    monkeypatch.delenv("PS1_FINAL_EVAL", raising=False)
    for split in ("final_external_test", "ood_test", "other_test"):
        with pytest.raises(PermissionError):
            select([], {split})
    assert select([{"split": "pv_val", "canonical": "x"}], {"pv_val"}) == [{"split": "pv_val", "canonical": "x"}]


def test_group_folds_never_split_a_group():
    from ps1.vision.dataset import group_folds
    groups = [i // 3 for i in range(90)]
    masks = group_folds(groups)
    assert sum(sum(m) for m in masks) == len(groups)                 # every row in exactly one fold
    for g in set(groups):
        rows = [i for i, x in enumerate(groups) if x == g]
        assert len({tuple(m[i] for m in masks) for i in rows}) == 1  # whole group on one side


def test_lab_ood_dev_becomes_ood_train_field_ood_stays_dev():
    rows = [row("plantvillage", "all", "tomato_leaf_mold", group=i) for i in range(40)]
    rows += [row("plantdoc", "train", "UNSUPPORTED", group=100 + i) for i in range(40)]
    assign(rows)
    pv, pd = {r["split"] for r in rows[:40]}, {r["split"] for r in rows[40:]}
    assert pv == {"ood_train", "ood_test"}
    assert pd == {"ood_dev", "ood_test"}


def test_audit_excludes_cross_split_pairs_the_lsh_misses():
    from ps1.data.build_splits import audit_near_duplicates
    # 6 differing bits spread over all four 16-bit bands: LSH never compares these, the audit must.
    a, b = 0, (1 << 0) | (1 << 17) | (1 << 34) | (1 << 51) | (1 << 5) | (1 << 22)
    rows = [dict(split="final_external_test", phash=f"{a:016x}", dhash="0" * 16, exclude_reason=""),
            dict(split="pv_train", phash=f"{b:016x}", dhash="0" * 16, exclude_reason=""),
            dict(split="pv_train", phash=f"{a:016x}", dhash="f" * 16, exclude_reason="")]   # dHash disagrees: kept
    assert audit_near_duplicates(rows) == 1
    assert rows[0]["split"] == "final_external_test"
    assert rows[1]["split"] == "excluded" and "audit" in rows[1]["exclude_reason"]
    assert rows[2]["split"] == "pv_train"
