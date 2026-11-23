"""W7 (outlier-exposure model): decision rule = argmax 'other' -> abstain, else accept if p >= t.

Out-of-fold only: each fold model scores the field_dev and PlantDoc-OOD halves it never trained on.
Maharashtra OOD (leaf miner, insects, disorder) was never trained on by any model, so it measures
behaviour on genuinely new problems. Sealed splits are never read.
Use the fixed-epoch last.pt checkpoints: best.pt is chosen on the very fold it would be scored on.
Novel Indian OOD images are scored by every fold model; their logits are averaged per image so each
photo counts once.
Usage: python -m ps1.vision.abstain_oe --fold-ckpts runs/v4_f0/last.pt runs/v4_f1/last.pt --out runs/v4
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from ps1.data.build_splits import OTHER_CLASS
from ps1.vision.abstain import fit_temperature
from ps1.vision.checkpoint import check_fold_checkpoints, ckpt_meta, load_fp32
from ps1.vision.dataset import eval_loader, input_size, read_manifest, select
from ps1.vision.train import as_other, predict, split_field, split_other

THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.78, 0.8, 0.9)   # 0.78: recommended gate (21 Nov)


def _logits(model: torch.nn.Module, rows: list[dict], classes: list[str], size: int) -> tuple[np.ndarray, np.ndarray]:
    return predict(model, eval_loader(rows, classes, size=size))


def collect(ckpts: list[str]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], dict]:
    """Out-of-fold logits per evaluation set: known field, seen-type field OOD, novel OOD (per image)."""
    meta = check_fold_checkpoints(ckpts)
    rows = read_manifest(taxonomy=meta["taxonomy"])
    size = input_size(meta["arch"])
    sets: dict[str, list] = {"known_field": [], "ood_field_seen_types": []}
    novel = as_other([r for r in select(rows, {"ood_dev"}) if r["source"] == "maharashtra"])
    novel_logits, novel_labels = [], None
    for fold, ckpt in enumerate(ckpts):
        model, classes = load_fp32(ckpt)
        _, known_held = split_field(rows, classes, fold, meta["fold_scheme"])
        _, _, ood_held = split_other(rows, fold, meta["fold_scheme"])
        sets["known_field"].append(_logits(model, known_held, classes, size))
        sets["ood_field_seen_types"].append(_logits(model, ood_held, classes, size))
        nl, novel_labels = _logits(model, novel, classes, size)
        novel_logits.append(nl)
    merged = {k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for k, v in sets.items()}
    merged["ood_novel_india"] = (np.mean(novel_logits, axis=0), novel_labels)   # one row per photo
    return merged, meta["classes"], meta


def decide(probs: np.ndarray, other_idx: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(accepted mask, predicted class). Accepted = top class is a disease and its probability >= threshold."""
    pred = probs.argmax(1)
    accepted = (pred != other_idx) & (probs.max(1) >= threshold)
    return accepted, pred


def evaluate(sets: dict, classes: list[str], temperature: float) -> dict:
    other = classes.index(OTHER_CLASS)
    known_idx = [i for i in range(len(classes)) if i != other]
    probs = {k: torch.from_numpy(v[0] / temperature).softmax(1).numpy() for k, v in sets.items()}
    known_y = sets["known_field"][1]
    out = {"n": {k: int(len(v[1])) for k, v in sets.items()},
           # Comparable with v2's 5-class field F1: predicting 'other' on a known image counts as an error.
           "known_field_macro_f1": float(f1_score(known_y, probs["known_field"].argmax(1), labels=known_idx,
                                                  average="macro", zero_division=0)),
           "known_sent_to_other": float((probs["known_field"].argmax(1) == other).mean()),
           "points": []}
    for t in THRESHOLDS:
        acc_k, pred_k = decide(probs["known_field"], other, t)
        point = {"threshold": t, "known_coverage": float(acc_k.mean()),
                 "accepted_error": float((pred_k[acc_k] != known_y[acc_k]).mean()) if acc_k.any() else None}
        for name in ("ood_field_seen_types", "ood_novel_india"):
            point[f"false_accept_{name}"] = float(decide(probs[name], other, t)[0].mean())
        out["points"].append(point)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets, classes, meta = collect(args.fold_ckpts)
    # Temperature on out-of-fold field logits (known + seen-type OOD): the deployment distribution.
    calib_logits = np.concatenate([sets["known_field"][0], sets["ood_field_seen_types"][0]])
    calib_labels = np.concatenate([sets["known_field"][1], sets["ood_field_seen_types"][1]])
    temperature = fit_temperature(calib_logits, calib_labels)
    result = {"temperature": temperature, "classes": classes, "taxonomy": meta["taxonomy"],
              "fold_scheme": meta["fold_scheme"], "epochs": [ckpt_meta(c)["epoch"] for c in args.fold_ckpts],
              "checkpoint_sha256": [hashlib.sha256(Path(c).read_bytes()).hexdigest() for c in args.fold_ckpts],
              "note": "dev estimate: temperature and thresholds are fit and scored on the same out-of-fold data",
              **evaluate(sets, classes, temperature)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "abstention_oe_dev.json").write_text(json.dumps(result, indent=1))

    print(f"T={temperature:.3f}  n={result['n']}")
    print(f"known field macro-F1 (known classes, 'other' = error) {result['known_field_macro_f1']:.3f}; "
          f"known sent to 'other' {result['known_sent_to_other']:.3f}")
    print(f"{'t':>4} {'coverage':>9} {'acc.err':>8} {'FA seen':>8} {'FA novel':>9}")
    for p in result["points"]:
        err = "  n/a" if p["accepted_error"] is None else f"{p['accepted_error']:.3f}"
        print(f"{p['threshold']:>4} {p['known_coverage']:>9.3f} {err:>8} "
              f"{p['false_accept_ood_field_seen_types']:>8.3f} {p['false_accept_ood_novel_india']:>9.3f}")


if __name__ == "__main__":
    main()
