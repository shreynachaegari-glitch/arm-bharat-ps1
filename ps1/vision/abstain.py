"""W7: abstention rule selection on dev data only (v4 2.5, 2.7; Experiment B).

1. Known-class scores = out-of-fold field logits: each field-fold model scores only the field_dev half
   it never trained on (the final model trains on all field_dev, so this is the honest field estimate).
2. Fit a temperature T on those field logits (NLL); score max-softmax(logits/T) and negative energy
   T*logsumexp(logits/T). Higher = more in-distribution for both.
3. Compare known-class field vs unsupported OOD dev (ood_dev, incl. Indian leaf-miner photos):
   AUROC, plus coverage / accepted-error / OOD false-accept at candidate thresholds.
The rule and threshold are chosen by a person from this table, then frozen; sealed splits are never read.
Usage: python -m ps1.vision.abstain --fold-ckpts runs/v2_f0/best.pt runs/v2_f1/best.pt --out runs/v2_final
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ps1.vision.checkpoint import check_fold_checkpoints, load_fp32
from ps1.vision.dataset import eval_loader, input_size, read_manifest, select
from ps1.vision.train import predict, split_field

TARGET_OOD_FALSE_ACCEPT = (0.05, 0.10, 0.20, 0.30)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    x, y = torch.from_numpy(logits).float(), torch.from_numpy(labels).long()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def scores(logits: np.ndarray, temperature: float) -> dict[str, np.ndarray]:
    z = torch.from_numpy(logits).float() / temperature
    return {"max_softmax": z.softmax(1).max(1).values.numpy(),
            "neg_energy": (temperature * torch.logsumexp(z, 1)).numpy()}


def operating_points(known: np.ndarray, known_correct: np.ndarray, ood: np.ndarray) -> list[dict]:
    """For each target OOD false-accept rate, the threshold and what it costs on known classes."""
    points = []
    for target in TARGET_OOD_FALSE_ACCEPT:
        threshold = float(np.quantile(ood, 1 - target))
        accepted = known >= threshold
        points.append({
            "target_ood_false_accept": target,
            "threshold": threshold,
            "ood_false_accept": float((ood >= threshold).mean()),
            "known_coverage": float(accepted.mean()),
            "accepted_error": float(1 - known_correct[accepted].mean()) if accepted.any() else None,
            "n_known": int(len(known)), "n_ood": int(len(ood)),
        })
    return points


def fold_logits(ckpts: list[str], ood_sources: set[str] | None = None,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Out-of-fold field logits (each fold model scores only the half it never trained on) + OOD-dev logits.

    Folds and labels come from the checkpoints' own metadata; OOD logits are averaged over fold models
    so each OOD image counts once.
    """
    meta = check_fold_checkpoints(ckpts)
    rows = read_manifest(taxonomy=meta["taxonomy"])
    field_logits, field_labels, ood_logits = [], [], []
    classes: list[str] = meta["classes"]
    for fold, ckpt in enumerate(ckpts):
        model, classes = load_fp32(ckpt)
        _, held_out = split_field(rows, classes, fold, meta["fold_scheme"])
        fl, fy = predict(model, eval_loader(held_out, classes, size=input_size(meta["arch"])))
        field_logits.append(fl)
        field_labels.append(fy)
        ood = [{**r, "canonical": classes[0]} for r in select(rows, {"ood_dev"})   # dummy label, unused
               if ood_sources is None or r["source"] in ood_sources]
        ood_logits.append(predict(model, eval_loader(ood, classes, size=input_size(meta["arch"])))[0])
    return np.concatenate(field_logits), np.concatenate(field_labels), np.mean(ood_logits, axis=0), classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-ckpts", nargs="+", required=True, help="fold-0 then fold-1 checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ood-sources", default="", help="comma list, e.g. plantdoc,maharashtra (field OOD only)")
    args = ap.parse_args()

    ood_sources = set(args.ood_sources.split(",")) if args.ood_sources else None
    field_logits, field_labels, ood_logits, classes = fold_logits(args.fold_ckpts, ood_sources)
    # Temperature on out-of-fold field logits: that is the distribution the farmer's photos come from.
    temperature = fit_temperature(field_logits, field_labels)
    correct = field_logits.argmax(1) == field_labels
    result = {"temperature": temperature, "classes": classes, "ood_sources": sorted(ood_sources or []),
              "n_field_oof": int(len(field_labels)),
              "n_ood_scores": int(len(ood_logits)), "field_oof_accuracy": float(correct.mean()), "methods": {}}
    for method in ("max_softmax", "neg_energy"):
        known = scores(field_logits, temperature)[method]
        ood = scores(ood_logits, temperature)[method]
        result["methods"][method] = {
            "auroc_known_vs_ood": float(roc_auc_score(np.r_[np.ones(len(known)), np.zeros(len(ood))], np.r_[known, ood])),
            "auroc_correct_vs_wrong": float(roc_auc_score(correct, known)) if 0 < correct.sum() < len(correct) else None,
            "operating_points": operating_points(known, correct, ood),
        }
    suffix = "_" + "_".join(sorted(ood_sources)) if ood_sources else ""
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, f"abstention_dev{suffix}.json").write_text(json.dumps(result, indent=1))

    print(f"temperature {temperature:.3f}; field OOF accuracy {correct.mean():.3f} (n={len(correct)}); OOD scores n={len(ood_logits)}")
    for method, r in result["methods"].items():
        cvw = r["auroc_correct_vs_wrong"]
        cvw_text = "n/a" if cvw is None else f"{cvw:.3f}"
        print(f"{method:12} AUROC known-vs-OOD {r['auroc_known_vs_ood']:.3f}  correct-vs-wrong {cvw_text}")
        for p in r["operating_points"]:
            err = "n/a" if p["accepted_error"] is None else f"{p['accepted_error']:.3f}"
            print(f"   OOD false-accept {p['ood_false_accept']:.2f} -> field coverage {p['known_coverage']:.3f}, accepted error {err}")


if __name__ == "__main__":
    main()
