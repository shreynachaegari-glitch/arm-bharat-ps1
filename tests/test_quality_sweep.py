import cv2
import numpy as np
import pytest

from ps1 import quality_gate
from ps1.contracts import QualityReason
from ps1.quality_sweep import degrade_blur, degrade_gain, load_grey, reason

MAP = {"ok": QualityReason.OK, "blurry": QualityReason.BLURRY, "too_dark": QualityReason.TOO_DARK,
       "too_bright": QualityReason.TOO_BRIGHT}


def _image(kind: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if kind == "sharp":
        return rng.integers(40, 200, (200, 200), dtype=np.uint8)
    if kind == "blurry":
        return np.full((200, 200), 120, dtype=np.uint8)
    if kind == "dark":
        return np.full((200, 200), 3, dtype=np.uint8)
    return np.full((200, 200), 254, dtype=np.uint8)


@pytest.mark.parametrize("kind", ["sharp", "blurry", "dark", "bright"])
def test_sweep_rule_agrees_with_the_real_gate(tmp_path, kind):
    path = tmp_path / f"{kind}.png"
    cv2.imwrite(str(path), _image(kind))
    t = quality_gate.QualityThresholds()
    gate = quality_gate.check(path, t)
    grey = load_grey(str(path))
    assert MAP[reason(*quality_gate.measure(grey), t)] == gate.reason


def test_blur_lowers_sharpness_and_gain_changes_brightness():
    grey = _image("sharp")
    assert quality_gate.measure(degrade_blur(4.0)(grey))[0] < quality_gate.measure(grey)[0]
    assert degrade_gain(0.3)(grey).mean() < grey.mean() < degrade_gain(2.5)(grey).mean()
