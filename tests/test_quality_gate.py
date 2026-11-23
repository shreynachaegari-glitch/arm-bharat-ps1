import cv2
import numpy as np
import pytest

from ps1.contracts import QualityReason
from ps1.quality_gate import QualityThresholds, check


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, size=(480, 640, 3), dtype=np.uint8)   # high-frequency, mid exposure
    return img


@pytest.fixture
def write(tmp_path):
    def _write(name: str, img: np.ndarray) -> str:
        path = tmp_path / name
        cv2.imwrite(str(path), img)
        return str(path)
    return _write


def test_sharp_well_exposed_passes(write):
    r = check(write("ok.png", textured()))
    assert r.ok and r.reason is QualityReason.OK


def test_blurred_image_rejected(write):
    blurred = cv2.GaussianBlur(textured(), (0, 0), sigmaX=8)
    r = check(write("blur.png", blurred))
    assert not r.ok and r.reason is QualityReason.BLURRY


def test_dark_image_rejected(write):
    r = check(write("dark.png", (textured() * 0.1).astype(np.uint8)))
    assert r.reason is QualityReason.TOO_DARK


def test_blown_out_image_rejected(write):
    r = check(write("bright.png", np.full((480, 640, 3), 255, np.uint8)))
    assert r.reason is QualityReason.TOO_BRIGHT


def test_unreadable_file_rejected(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert check(bad).reason is QualityReason.UNREADABLE


def test_dark_shadowed_photo_reported_as_dark_not_bright(write):
    img = np.random.default_rng(1).integers(170, 250, size=(480, 640, 3), dtype=np.uint8)
    img[:, :480] = 0                     # 75% pure black (deep shadow, > q1's 0.70), mean grey still >= 40
    r = check(write("shadow.png", img))
    assert r.reason is QualityReason.TOO_DARK


def test_leaf_on_white_paper_passes(write):
    """Sreyesh 21 Nov (q1): a well-exposed leaf held against white paper is not a bad photo (q0 rejected 7.2% of dev)."""
    img = textured()
    img[:, :320] = 255                   # half the frame is blown-out white paper
    r = check(write("paper.png", img))
    assert r.ok and r.reason is QualityReason.OK


def test_thresholds_are_the_confirmed_q1():
    t = QualityThresholds()
    assert t.version == "q1-dev" and "placeholder" not in t.version
    assert t.max_clipped_fraction == 0.70
    assert (t.min_sharpness, t.min_mean, t.max_mean) == (60.0, 40.0, 220.0)   # unchanged from q0
