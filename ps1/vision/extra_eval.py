"""Score a deployed model on the held-out extra source (NEU, Bangladesh field; data/manifest_extra.csv, 21 Nov).

Same app path as the India check (sidecar-checked .pte -> app preprocessing -> logits -> bound decision config).
Known classes: x_test_known (early blight, late blight, healthy). Unsupported: x_test_ood (leaf curl, bushy stunt).
The held-out source is never in training or selection, so v7 (trained before it was downloaded) and later models
are compared on equal terms. Reports the deployed threshold plus a few others (same temperature).
22 Nov: the held-out source is split into a dev half (choosing v9 variants) and a test half; --part picks which.
'all' (both halves) is what v7/v8 were scored on on 21 Nov and keeps that file name.
Usage: python -m ps1.vision.extra_eval --pte <model>.pte --decision <decision>.json --out <dir> [--part test|dev|all]
"""
import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from ps1.data.build_splits import OTHER_CLASS
from ps1.decision import DecisionConfig, decide
from ps1.vision.dataset import REPO_ROOT, read_manifest
from ps1.vision.runtime import load_vision

EXTRA = REPO_ROOT / "data/manifest_extra.csv"
THRESHOLDS = (0.0, 0.6, 0.7, 0.78, 0.8, 0.9)
HEALTHY = "tomato_healthy"
Z = 1.96
PARTS = ("test", "dev", "all")


def splits_for(part: str) -> tuple[set[str], set[str]]:
    """(known splits, unsupported splits) of the held-out source for one part."""
    halves = {"test": ["x_test"], "dev": ["x_dev"], "all": ["x_test", "x_dev"]}[part]
    return {f"{h}_known" for h in halves}, {f"{h}_ood" for h in halves}


def out_name(part: str) -> str:
    return "extra_eval.json" if part == "all" else f"extra_eval_{part}.json"


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre, half = p + Z * Z / (2 * n), Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return [max(0.0, (centre - half) / (1 + Z * Z / n)), min(1.0, (centre + half) / (1 + Z * Z / n))]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def macro_f1(truth: list[str], pred: list[str], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(truth, pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(truth, pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(truth, pred, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s))


def score(known: list[tuple[str, np.ndarray]], ood: list[np.ndarray], cfg: DecisionConfig, classes: tuple[str, ...],
          sha: str) -> dict:
    kd = [(t, decide(z, cfg, classes, sha)) for t, z in known]
    od = [decide(z, cfg, classes, sha) for z in ood]
    answered = [(t, d.label) for t, d in kd if d.label is not None]
    sick = [(t, d) for t, d in kd if t != HEALTHY]
    return {"threshold": cfg.threshold,
            "coverage": rate(len(answered), len(kd)),
            "accepted_error": rate(sum(t != p for t, p in answered), len(answered)),
            "sick_called_healthy": rate(sum(d.label == HEALTHY for _, d in sick), len(sick)),
            "false_accept_ood": rate(sum(d.label is not None for d in od), len(od)),
            "ood_called_healthy": rate(sum(d.label == HEALTHY for d in od), len(od))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=PARTS, default="test")
    args = ap.parse_args()
    vision, cfg = load_vision(args.pte), DecisionConfig.load(args.decision)
    rows = read_manifest(EXTRA, taxonomy=vision.taxonomy)
    known_splits, ood_splits = splits_for(args.part)
    known_rows = [r for r in rows if r["split"] in known_splits]
    ood_rows = [r for r in rows if r["split"] in ood_splits]
    if not known_rows or not ood_rows:
        raise SystemExit(f"no held-out rows for part {args.part!r} in {EXTRA}")
    logits = {r["path"]: np.asarray(vision.logits(vision.preprocess(r["path"]))) for r in known_rows + ood_rows}
    known = [(r["canonical"], logits[r["path"]]) for r in known_rows]
    ood = [logits[r["path"]] for r in ood_rows]
    supported = [c for c in vision.classes if c != OTHER_CLASS]
    present = sorted({t for t, _ in known})
    argmax = [vision.classes[int(np.argmax(z))] for _, z in known]       # forced choice, no abstention
    results = {
        "pte_sha256": vision.pte_sha256, "decision": cfg.version, "temperature": cfg.temperature,
        "extra_manifest": str(EXTRA.relative_to(REPO_ROOT)), "part": args.part, "n_known": dict(Counter(t for t, _ in known)),
        "n_ood": dict(Counter(r["source_label"] for r in ood_rows)),
        "forced_choice_macro_f1_present_classes": macro_f1([t for t, _ in known], argmax, present),
        "forced_choice_confusion": {t: dict(Counter(p for (tt, _), p in zip(known, argmax, strict=True) if tt == t))
                                    for t in present},
        "supported_classes": supported,
        "deployed": score(known, ood, cfg, vision.classes, vision.pte_sha256),
        "curve": [score(known, ood, replace(cfg, threshold=t), vision.classes, vision.pte_sha256) for t in THRESHOLDS],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / out_name(args.part)).write_text(json.dumps(results, indent=1))
    d = results["deployed"]
    print(f"{args.pte} [{args.part}]: F1(forced, {len(present)} classes) {results['forced_choice_macro_f1_present_classes']:.3f}; "
          f"t={d['threshold']}: answers {d['coverage']['k']}/{d['coverage']['n']}, wrong "
          f"{d['accepted_error']['k']}/{d['accepted_error']['n']}, sick->healthy {d['sick_called_healthy']['k']}/"
          f"{d['sick_called_healthy']['n']}, unsupported accepted {d['false_accept_ood']['k']}/{d['false_accept_ood']['n']}")


if __name__ == "__main__":
    main()
