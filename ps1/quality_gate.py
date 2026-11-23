"""Quality gate: reject blurry, dark or blown-out photos before any model runs (v4 section 4).

Works on a downsampled greyscale copy so it costs a few ms on the Pi.
Thresholds q1-dev (21 Nov 2026; evidence: Gate Decision Sheet 2026-11-04 decision 3); `version` is written
into every Finding so a result can be traced to the thresholds that made it.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ps1.contracts import QualityReason

ANALYSIS_MAX_SIDE = 256          # px; blur score is resolution-dependent, so always measure at this size


@dataclass(frozen=True)
class QualityThresholds:
    version: str = "q1-dev"
    min_sharpness: float = 60.0      # variance of the Laplacian
    min_mean: float = 40.0           # mean grey level, 0-255
    max_mean: float = 220.0
    max_clipped_fraction: float = 0.70   # share crushed to black (<=5) or blown to white (>=250); q0 0.25 rejected
                                         # 7.2% of good field-dev photos (leaves on white/black backgrounds), q1 1.5%


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: QualityReason
    sharpness: float
    mean: float
    dark_clipped: float
    bright_clipped: float
    thresholds_version: str


def _load_grey(image_path: str | Path) -> np.ndarray | None:
    grey = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return None
    scale = ANALYSIS_MAX_SIDE / max(grey.shape)
    if scale < 1.0:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey


def measure(grey: np.ndarray) -> tuple[float, float, float, float]:
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return sharpness, float(grey.mean()), float(np.mean(grey <= 5)), float(np.mean(grey >= 250))


def check(image_path: str | Path, t: QualityThresholds = QualityThresholds()) -> QualityResult:
    grey = _load_grey(image_path)
    if grey is None:
        return QualityResult(False, QualityReason.UNREADABLE, 0.0, 0.0, 0.0, 0.0, t.version)
    sharpness, mean, dark, bright = measure(grey)
    if mean < t.min_mean or dark > t.max_clipped_fraction:
        reason = QualityReason.TOO_DARK
    elif mean > t.max_mean or bright > t.max_clipped_fraction:
        reason = QualityReason.TOO_BRIGHT
    elif sharpness < t.min_sharpness:
        reason = QualityReason.BLURRY
    else:
        reason = QualityReason.OK
    return QualityResult(reason is QualityReason.OK, reason, sharpness, mean, dark, bright, t.version)
