import hashlib

import numpy as np

from ps1.vision.golden import compare, entry, sealed_overlap


def _entry(logits: list[float], tensor_sha: str = "a" * 64) -> dict:
    return entry("x.jpg", np.asarray(logits, dtype=np.float32), tensor_sha)


def test_identical_logits_pass():
    golden = [_entry([1.0, 2.0, 0.5])]
    report = compare(golden, [_entry([1.0, 2.0, 0.5])], tolerance=1e-3)
    assert report["passed"] and report["top1_agreement"] == 1.0 and report["max_abs_diff"] == 0.0


def test_small_float_noise_passes_within_tolerance():
    report = compare([_entry([1.0, 2.0])], [_entry([1.0005, 2.0])], tolerance=1e-3)
    assert report["passed"]


def test_logit_drift_above_tolerance_fails_even_with_same_top1():
    report = compare([_entry([1.0, 2.0])], [_entry([1.2, 2.0])], tolerance=1e-3)
    assert not report["passed"] and report["top1_agreement"] == 1.0
    assert report["worst_image"] == "x.jpg"


def test_top1_flip_fails():
    report = compare([_entry([1.0, 2.0])], [_entry([2.0, 1.0])], tolerance=5.0)
    assert not report["passed"] and report["top1_agreement"] == 0.0


def test_preprocess_difference_is_reported_separately():
    # Different JPEG decoder on the Pi: tensors differ, but that is a different finding from a runtime difference.
    report = compare([_entry([1.0, 2.0], "a" * 64)], [_entry([1.0, 2.0], "b" * 64)], tolerance=1e-3)
    assert report["passed"] and report["tensor_mismatches"] == 1


def test_image_sets_must_match():
    other = entry("y.jpg", np.asarray([1.0, 2.0], dtype=np.float32), "a" * 64)
    report = compare([_entry([1.0, 2.0])], [other], tolerance=1e-3)
    assert not report["passed"] and report["missing"] == ["x.jpg"] and report["extra"] == ["y.jpg"]


def test_a_sealed_image_is_detected_by_its_bytes_whatever_its_name(tmp_path):
    image = tmp_path / "harmless_name.jpg"
    image.write_bytes(b"leaf")
    other = tmp_path / "other.jpg"
    other.write_bytes(b"different")
    listing = tmp_path / "sealed.sha256"
    listing.write_text(f"ood_test  {hashlib.sha256(b'leaf').hexdigest()}  data/some where/x.jpg\n")
    assert sealed_overlap([image, other], listing) == ["harmless_name.jpg"]


def test_no_listing_means_no_overlap_check_on_the_pi(tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"leaf")
    assert sealed_overlap([image], tmp_path / "absent") == []
