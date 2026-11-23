"""22 Nov: NEU-dev scoring of out-of-fold checkpoints (v9 screening) uses the app's decision rule."""
import numpy as np

from ps1.vision.neu_dev_eval import fold_score, pooled

CLASSES = ["tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_spot", "tomato_other_problem"]


def onehot(i: int, strength: float = 10.0) -> np.ndarray:
    z = np.zeros(5)
    z[i] = strength
    return z


def test_fold_score_reports_forced_f1_and_decisions_at_the_threshold():
    truth = ["tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_late_blight"]
    logits = np.stack([onehot(0), onehot(1), onehot(2), onehot(0, 0.1)])   # last: forced wrong, abstains
    s = fold_score(logits, truth, np.stack([onehot(4), onehot(1)]), CLASSES, temperature=1.0)
    assert s["threshold"] == 0.8
    assert (s["coverage"]["k"], s["accepted_error"]["k"]) == (3, 0)
    assert s["false_accept_ood"]["k"] == 1 and s["ood_called_healthy"]["k"] == 1
    assert abs(s["f1_forced"] - np.mean([2 / 3, 1.0, 2 / 3])) < 1e-9


def test_pooled_sums_counts_and_averages_f1_over_folds():
    a = {"f1_forced": 0.6, "coverage": {"k": 1, "n": 4}, "accepted_error": {"k": 0, "n": 1},
         "sick_called_healthy": {"k": 0, "n": 3}, "false_accept_ood": {"k": 1, "n": 2}, "ood_called_healthy": {"k": 0, "n": 2}}
    b = {"f1_forced": 0.8, "coverage": {"k": 3, "n": 4}, "accepted_error": {"k": 1, "n": 3},
         "sick_called_healthy": {"k": 1, "n": 3}, "false_accept_ood": {"k": 0, "n": 2}, "ood_called_healthy": {"k": 0, "n": 2}}
    p = pooled([a, b])
    assert abs(p["f1_forced"] - 0.7) < 1e-9
    assert p["coverage"] == 0.5 and p["accepted_error"] == 0.25 and p["sick_called_healthy"] == 1 / 6
    assert p["false_accept_ood"] == 0.25
