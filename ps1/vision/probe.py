"""Frozen-backbone linear probes: cheap ranking of backbones and of field-data use (W4 triage).

Features are extracted once per (backbone, split) and cached. Two protocols, both on dev data only:
  A  logistic regression on PV train (+ ext_train) -> score on all of field_dev
  B  2-fold group-aware cross-validation inside field_dev: train on PV/ext train + one field fold,
     score on the other fold, average. Folds follow the dedup groups, so near-duplicates never
     sit on both sides. The sealed test is never read.
Usage: python -m ps1.vision.probe --backbone convnext_tiny_imagenet --out runs/probe
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v3_small,
)

from ps1.data.build_splits import SUPPORTED
from ps1.vision.dataset import (
    ManifestDataset,
    eval_transform,
    group_folds,
    read_manifest,
    select,
)

CLASSES = list(SUPPORTED)
PV_PER_CLASS = 400            # subsample PV train so lab images don't swamp the few field images
SPLITS = ("pv_train", "ext_train", "field_dev")


def load_backbone(name: str) -> torch.nn.Module:
    if name == "mnv3s_imagenet":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        return model.eval()
    # Weights-only backbones bundled with torchvision (no third-party code is fetched or executed).
    if name == "effnet_b0_imagenet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        return model.eval()
    if name == "convnext_tiny_imagenet":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = torch.nn.Identity()
        return model.eval()
    raise ValueError(name)


@torch.no_grad()
def features(model: torch.nn.Module, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ManifestDataset(rows, CLASSES, eval_transform()), batch_size=32, num_workers=8)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x).numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def cached(out: Path, backbone: str, split: str, rows: list[dict], model_fn) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Key the cache on the exact rows, so a rebuilt manifest can never reuse stale features.
    rows_key = hashlib.sha256("\n".join(f"{r['path']}|{r['canonical']}" for r in rows).encode()).hexdigest()[:12]
    path = out / f"feat_{backbone}_{split}_{rows_key}.npz"
    groups = [r["group"] for r in rows]
    if not path.exists():
        f, y = features(model_fn(), rows)
        np.savez(path, f=f, y=y)
    data = np.load(path)
    return data["f"], data["y"], groups


def probe(train_f: np.ndarray, train_y: np.ndarray, test_f: np.ndarray, test_y: np.ndarray) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
    clf.fit(train_f, train_y)
    return float(f1_score(test_y, clf.predict(test_f), labels=list(range(len(CLASSES))), average="macro"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", default="runs/probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)

    rows = read_manifest()
    rng = random.Random(0)
    pv = select(rows, {"pv_train"}, CLASSES)
    pv = [r for c in CLASSES for r in rng.sample([r for r in pv if r["canonical"] == c],
                                                min(PV_PER_CLASS, sum(r["canonical"] == c for r in pv)))]
    split_rows = {"pv_train": pv, "ext_train": select(rows, {"ext_train"}, CLASSES),
                  "field_dev": select(rows, {"field_dev"}, CLASSES)}

    model_holder: dict[str, torch.nn.Module] = {}

    def model_fn() -> torch.nn.Module:
        return model_holder.setdefault("m", load_backbone(args.backbone))

    feats = {s: cached(out, args.backbone, s, split_rows[s], model_fn) for s in SPLITS}
    base_f = np.concatenate([feats["pv_train"][0], feats["ext_train"][0]])
    base_y = np.concatenate([feats["pv_train"][1], feats["ext_train"][1]])
    field_f, field_y, field_groups = feats["field_dev"]

    result = {"backbone": args.backbone, "feature_dim": int(base_f.shape[1]),
              "A_lab_only_field_f1": probe(base_f, base_y, field_f, field_y)}
    fold_scores = []
    field_labels = [str(int(y)) for y in field_y]   # stratify like training does
    for mask in group_folds(field_groups, labels=field_labels):
        test_mask = np.array(mask)
        train_f = np.concatenate([base_f, field_f[~test_mask]])
        train_y = np.concatenate([base_y, field_y[~test_mask]])
        fold_scores.append(probe(train_f, train_y, field_f[test_mask], field_y[test_mask]))
    result["B_with_field_folds_f1"] = float(np.mean(fold_scores))
    result["B_fold_scores"] = fold_scores
    (out / f"probe_{args.backbone}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
