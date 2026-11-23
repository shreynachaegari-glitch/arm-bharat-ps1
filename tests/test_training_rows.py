import argparse

from ps1.data.build_splits import OTHER_CLASS, SUPPORTED
from ps1.vision.train import training_rows


def r(split, source, canonical, group):
    return dict(split=split, source=source, canonical=canonical, source_canonical=canonical, group=str(group),
                path=f"{source}/{group}.jpg")


ROWS = (
    [r("pv_train", "plantvillage", "tomato_early_blight", i) for i in range(10)]
    + [r("field_dev", "plantdoc", "tomato_early_blight", 100 + i) for i in range(20)]
    + [r("ood_train", "plantvillage", "tomato_leaf_mold", 200 + i) for i in range(10)]
    + [r("ood_dev", "plantdoc", "UNSUPPORTED", 300 + i) for i in range(20)]
    + [r("ood_dev", "maharashtra", "UNSUPPORTED", 400 + i) for i in range(5)]
    + [r("final_external_test", "plantdoc", "tomato_early_blight", 500 + i) for i in range(5)]
)
CLASSES = [*SUPPORTED, OTHER_CLASS]


def test_folds_keep_train_and_selection_disjoint_and_skip_maharashtra():
    for fold in (0, 1):
        train, held = training_rows(ROWS, CLASSES, argparse.Namespace(field_fold=fold, field_oversample=3, fold_scheme="stratified"))
        assert not {x["group"] for x in train} & {x["group"] for x in held}
        assert all(x["source"] != "maharashtra" for x in train + held)
        assert all(x["split"] != "final_external_test" for x in train + held)
        assert any(x["canonical"] == OTHER_CLASS for x in held)


def test_final_run_uses_all_field_rows_and_no_selection_set():
    train, held = training_rows(ROWS, CLASSES, argparse.Namespace(field_fold=-1, field_oversample=1, fold_scheme="stratified"))
    assert held == []
    assert sum(x["source"] == "plantdoc" for x in train) == 40          # 20 field_dev + 20 field OOD


def test_without_other_class_no_ood_rows_are_used():
    train, held = training_rows(ROWS, list(SUPPORTED), argparse.Namespace(field_fold=0, field_oversample=1, fold_scheme="stratified"))
    assert all(x["canonical"] != OTHER_CLASS for x in train + held)


def test_stratified_folds_balance_classes_and_keep_groups_whole():
    from collections import Counter

    from ps1.vision.dataset import group_folds
    labels = ["a"] * 40 + ["b"] * 12                      # rare class b
    groups = [str(i // 2) for i in range(52)]             # pairs of near-duplicates
    masks = group_folds(groups, labels=labels)
    for m in masks:
        held = Counter(l for l, h in zip(labels, m) if h)
        assert held["b"] == 6 and held["a"] == 20         # rare class split evenly
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        assert len({tuple(m[i] for m in masks) for i in idx}) == 1


def test_t2_taxonomy_merges_leaf_spots_but_keeps_original_label(tmp_path):
    import csv

    from ps1.vision.dataset import read_manifest
    path = tmp_path / "m.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "canonical", "split"])
        w.writeheader()
        w.writerows([{"path": "a.jpg", "canonical": "tomato_bacterial_spot", "split": "pv_train"},
                     {"path": "b.jpg", "canonical": "tomato_early_blight", "split": "pv_train"}])
    rows = read_manifest(path, taxonomy="t2")
    assert [r["canonical"] for r in rows] == ["tomato_leaf_spot", "tomato_early_blight"]
    assert rows[0]["source_canonical"] == "tomato_bacterial_spot"
    assert read_manifest(path)[0]["canonical"] == "tomato_bacterial_spot"     # t1 default unchanged


def test_composite_keeps_dark_lesions_inside_the_leaf():
    import random

    import numpy as np
    from PIL import Image

    from ps1.vision.dataset import composite_on_background
    leaf = np.zeros((64, 64, 3), np.uint8)
    leaf[8:56, 8:56] = (40, 160, 40)                 # green leaf on black
    leaf[28:36, 28:36] = (5, 5, 5)                   # dark lesion, below the mask threshold
    bg = Image.fromarray(np.full((128, 128, 3), (250, 250, 250), np.uint8))
    out = np.asarray(composite_on_background(Image.fromarray(leaf), bg, random.Random(0)))
    assert out[32, 32].max() < 60                    # lesion kept, not replaced by the bright background
    assert out[2, 2].min() > 200                     # outside the leaf is background


def test_exclude_sources_drops_only_that_source_from_training():
    rows = ROWS + [r("ext_train", "maharashtra", "tomato_early_blight", 900 + i) for i in range(4)]
    keep = argparse.Namespace(field_fold=0, field_oversample=1, fold_scheme="stratified")
    drop = argparse.Namespace(field_fold=0, field_oversample=1, fold_scheme="stratified", exclude_sources="maharashtra")
    with_ext, _ = training_rows(rows, CLASSES, keep)
    without_ext, held = training_rows(rows, CLASSES, drop)
    assert sum(x["source"] == "maharashtra" for x in with_ext) == 4
    assert not any(x["source"] == "maharashtra" for x in without_ext)
    assert len(with_ext) - len(without_ext) == 4


def test_extra_rows_add_field_and_other_training_but_never_selection_or_held_out_test():
    """21 Nov: extra field sources (manifest_extra.csv) train only; their held-out source is never read here."""
    extra = ([r("x_field_train", "bd_diu_2025a", "tomato_early_blight", f"x{i}") for i in range(6)]
             + [r("x_ood_train", "bd_diu_2025a", "UNSUPPORTED", f"y{i}") for i in range(4)]
             + [r("x_test_known", "bd_neu_2026", "tomato_early_blight", f"z{i}") for i in range(3)]
             + [r("x_test_ood", "bd_neu_2026", "UNSUPPORTED", f"w{i}") for i in range(3)])
    base = argparse.Namespace(field_fold=0, field_oversample=1, fold_scheme="stratified")
    plus = argparse.Namespace(field_fold=0, field_oversample=1, fold_scheme="stratified", extra_oversample=2)
    train0, held0 = training_rows(ROWS, CLASSES, base)
    train1, held1 = training_rows(ROWS + extra, CLASSES, plus)
    added = [x for x in train1 if x["source"] == "bd_diu_2025a"]
    assert len(added) == 2 * 10                                   # 6 disease + 4 'other', oversampled x2
    assert sum(x["canonical"] == OTHER_CLASS for x in added) == 8
    assert held1 == held0                                         # selection set unchanged: comparable to v7
    assert not any(x["source"] == "bd_neu_2026" for x in train1 + held1)
    assert len(train1) - len(train0) == 20


def test_extra_rows_ignored_without_the_flag():
    extra = [r("x_field_train", "bd_diu_2025a", "tomato_early_blight", f"x{i}") for i in range(6)]
    args = argparse.Namespace(field_fold=0, field_oversample=1, fold_scheme="stratified")
    assert training_rows(ROWS + extra, CLASSES, args)[0] == training_rows(ROWS, CLASSES, args)[0]
