"""W3 evidence: how the quality gate behaves on real photos and on deliberately degraded copies (dev only).

Two questions for one set of thresholds: how many acceptable real field photos does it wrongly reject, and how
many ruined copies does it catch? Dataset photos are already curated, so they stand in for "acceptable". The
degraded copies are synthetic, so they estimate catch rates only roughly; real bad phone photos are still better.
Sealed splits are never read. Usage: python -m ps1.quality_sweep --out runs/quality
"""
import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ps1.quality_gate import ANALYSIS_MAX_SIDE, QualityThresholds, measure
from ps1.vision.dataset import read_manifest, select

GROUPS = {"plantdoc_field_dev": ("field_dev", "plantdoc"), "plantdoc_ood_dev": ("ood_dev", "plantdoc"),
          "maharashtra_ood_dev": ("ood_dev", "maharashtra"), "maharashtra_ext": ("ext_train", "maharashtra")}
BLUR_SIGMAS = (2.0, 4.0, 8.0)      # px at ANALYSIS_MAX_SIDE; 2 is soft, 8 shows no lesion detail
DARK_GAINS = (0.3, 0.15)           # multiply brightness
BRIGHT_GAINS = (2.5, 4.0)


def degrade_blur(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda grey: cv2.GaussianBlur(grey, (0, 0), sigma)


def degrade_gain(gain: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda grey: np.clip(grey.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def reason(sharpness: float, mean: float, dark: float, bright: float, t: QualityThresholds) -> str:
    """Same order of checks as quality_gate.check; a test holds the two together."""
    if mean < t.min_mean or dark > t.max_clipped_fraction:
        return "too_dark"
    if mean > t.max_mean or bright > t.max_clipped_fraction:
        return "too_bright"
    return "blurry" if sharpness < t.min_sharpness else "ok"


def load_grey(path: str) -> np.ndarray | None:
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return None
    scale = ANALYSIS_MAX_SIDE / max(grey.shape)
    return cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else grey


def reject_rate(greys: list[np.ndarray], t: QualityThresholds) -> dict:
    counts = Counter(reason(*measure(g), t) for g in greys)
    return {"n": len(greys), "rejected": (len(greys) - counts["ok"]) / len(greys), "reasons": dict(counts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clipped", type=float, help="try another clipping limit without changing the gate")
    args = ap.parse_args()
    t = QualityThresholds()
    if args.max_clipped is not None:
        t = replace(t, version=f"sweep-clip{args.max_clipped}", max_clipped_fraction=args.max_clipped)
    manifest = read_manifest(taxonomy="t2")
    result: dict = {"thresholds": t.__dict__, "real": {}, "synthetic": {}, "sharpness_quantiles": {}}
    all_greys: list[np.ndarray] = []
    for name, (split, source) in GROUPS.items():
        rows = [r for r in select(manifest, {split}) if r["source"] == source]
        greys = [g for g in (load_grey(r["path"]) for r in rows) if g is not None]
        all_greys += greys
        result["real"][name] = reject_rate(greys, t)
        sharp = [measure(g)[0] for g in greys]
        result["sharpness_quantiles"][name] = {f"p{q}": float(np.percentile(sharp, q)) for q in (1, 5, 10, 25, 50)}
    # Degrade only photos the gate accepts: a catch must come from the damage, not from the photo already being bad.
    good = [g for g in all_greys if reason(*measure(g), t) == "ok"]
    for label, fns in (("blur_sigma", {s: degrade_blur(s) for s in BLUR_SIGMAS}),
                       ("dark_gain", {g: degrade_gain(g) for g in DARK_GAINS}),
                       ("bright_gain", {g: degrade_gain(g) for g in BRIGHT_GAINS})):
        for value, fn in fns.items():
            result["synthetic"][f"{label}_{value}"] = reject_rate([fn(g) for g in good], t)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"quality_sweep_{t.version}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
