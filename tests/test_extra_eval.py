"""21 Nov: held-out extra-source scoring uses the deployed decision rule."""
import numpy as np

from ps1.decision import DecisionConfig
from ps1.vision.extra_eval import macro_f1, out_name, score, splits_for, wilson

CLASSES = ("tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_spot", "tomato_other_problem")
CFG = DecisionConfig("d", "sha", CLASSES, 1.0, 0.8, "test")


def onehot(i: int, strength: float = 10.0) -> np.ndarray:
    z = np.zeros(5)
    z[i] = strength
    return z


def test_score_counts_answers_errors_sick_as_healthy_and_false_accepts():
    known = [("tomato_early_blight", onehot(0)), ("tomato_early_blight", onehot(1)),   # right; sick called healthy
             ("tomato_late_blight", onehot(2, 0.1)), ("tomato_healthy", onehot(1))]    # low confidence -> abstain
    ood = [onehot(4), onehot(1), onehot(0, 0.1)]                                         # other; healthy; abstain
    s = score(known, ood, CFG, CLASSES, "sha")
    assert (s["coverage"]["k"], s["coverage"]["n"]) == (3, 4)
    assert (s["accepted_error"]["k"], s["accepted_error"]["n"]) == (1, 3)
    assert (s["sick_called_healthy"]["k"], s["sick_called_healthy"]["n"]) == (1, 3)
    assert (s["false_accept_ood"]["k"], s["ood_called_healthy"]["k"]) == (1, 1)


def test_wilson_and_macro_f1():
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0


def test_parts_select_dev_test_or_both_and_keep_the_v8_file_name_for_all():
    assert splits_for("test") == ({"x_test_known"}, {"x_test_ood"})
    assert splits_for("dev") == ({"x_dev_known"}, {"x_dev_ood"})
    assert splits_for("all") == ({"x_test_known", "x_dev_known"}, {"x_test_ood", "x_dev_ood"})
    assert out_name("all") == "extra_eval.json" and out_name("test") == "extra_eval_test.json"
