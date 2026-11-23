"""NEU-dev score of out-of-fold checkpoints, for choosing fine-tuning variants (v9, 22 Nov).

Sreyesh (22 Nov): half of the held-out NEU source (x_dev_known / x_dev_ood in data/manifest_extra.csv) is used to
choose; the other half (x_test_*) stays untouched until the final check. Neither fold model trained on NEU, so each
is scored on all of NEU-dev; counts are pooled over the two folds and forced-choice F1 is averaged. The decision is
the app's (ps1.decision.decide) at threshold 0.80 with T from the same seed's PlantDoc out-of-fold fit.
Usage: python -m ps1.vision.neu_dev_eval --fold-ckpts <f0>/ema.pt <f1>/ema.pt --abstention <dir>/abstention_oe_dev.json
       --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ps1.decision import DecisionConfig
from ps1.vision.checkpoint import check_fold_checkpoints, load_fp32
from ps1.vision.dataset import eval_loader, input_size, read_manifest
from ps1.vision.extra_eval import EXTRA, macro_f1, score, splits_for
from ps1.vision.train import as_other, predict

THRESHOLD = 0.8
RATES = ("coverage", "accepted_error", "sick_called_healthy", "false_accept_ood", "ood_called_healthy")
FOLD_SHA = "fold-checkpoint"          # decide() binds a config to a model; here both are the in-memory fold model


def fold_score(known_logits: np.ndarray, known_truth: list[str], ood_logits: np.ndarray, classes: list[str],
               temperature: float, threshold: float = THRESHOLD) -> dict:
    cfg = DecisionConfig("neu-dev", FOLD_SHA, tuple(classes), temperature, threshold, "PlantDoc out-of-fold T")
    s = score(list(zip(known_truth, known_logits, strict=True)), list(ood_logits), cfg, tuple(classes), FOLD_SHA)
    forced = [classes[int(np.argmax(z))] for z in known_logits]
    s["f1_forced"] = macro_f1(known_truth, forced, sorted(set(known_truth)))
    return s


def pooled(folds: list[dict]) -> dict:
    out = {"f1_forced": float(np.mean([f["f1_forced"] for f in folds]))}
    for name in RATES:
        k, n = sum(f[name]["k"] for f in folds), sum(f[name]["n"] for f in folds)
        out[name] = k / n if n else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-ckpts", nargs="+", required=True)
    ap.add_argument("--abstention", required=True, help="abstention_oe_dev.json of the same seed (temperature)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = check_fold_checkpoints(args.fold_ckpts)
    temperature = float(json.loads(Path(args.abstention).read_text())["temperature"])
    rows = read_manifest(EXTRA, taxonomy=meta["taxonomy"])
    known_splits, ood_splits = splits_for("dev")
    known = [r for r in rows if r["split"] in known_splits]
    ood = as_other([r for r in rows if r["split"] in ood_splits])
    if not known or not ood:
        raise SystemExit(f"no NEU-dev rows in {EXTRA}")
    folds = []
    for ckpt in args.fold_ckpts:
        model, classes = load_fp32(ckpt)
        size = input_size(meta["arch"])
        kl, _ = predict(model, eval_loader(known, classes, size=size))
        ol, _ = predict(model, eval_loader(ood, classes, size=size))
        folds.append({"ckpt": ckpt, **fold_score(kl, [r["canonical"] for r in known], ol, classes, temperature)})
    result = {"temperature": temperature, "threshold": THRESHOLD, "n_known": len(known), "n_ood": len(ood),
              "pooled": pooled(folds), "folds": folds}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "neu_dev.json").write_text(json.dumps(result, indent=1))
    p = result["pooled"]
    print(f"NEU-dev ({len(known)} known, {len(ood)} unsupported): F1(forced) {p['f1_forced']:.3f}; t={THRESHOLD}: "
          f"answers {p['coverage']:.3f}, error {p['accepted_error']}, sick->healthy {p['sick_called_healthy']:.3f}, "
          f"unsupported accepted {p['false_accept_ood']:.3f}")


if __name__ == "__main__":
    main()
