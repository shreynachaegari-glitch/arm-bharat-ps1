import numpy as np

from ps1.vision.abstain import fit_temperature, operating_points, scores


def test_temperature_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 400)
    logits = rng.normal(size=(400, 3)) * 2
    logits[np.arange(400), labels] += 1.0
    logits *= 5                                   # same ranking, far too confident
    assert fit_temperature(logits.astype(np.float32), labels) > 2.0


def test_scores_rank_confident_rows_higher():
    logits = np.array([[10.0, 0.0, 0.0], [0.3, 0.2, 0.1]], dtype=np.float32)
    for s in scores(logits, 1.0).values():
        assert s[0] > s[1]


def test_operating_points_respect_target_false_accept():
    rng = np.random.default_rng(1)
    ood = rng.uniform(0, 0.6, 1000)
    known = rng.uniform(0.4, 1.0, 1000)
    correct = known > 0.5
    for p in operating_points(known, correct, ood):
        assert abs(p["ood_false_accept"] - p["target_ood_false_accept"]) < 0.02
        assert 0 < p["known_coverage"] <= 1
