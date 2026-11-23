import numpy as np
import pytest

from ps1.vision.final_eval import OTHER_CLASS
from ps1.vision.gate_sheet import (
    METRICS,
    add_pass_probabilities,
    load_limits,
    pass_probability,
    sheet_row,
    wilson,
)

CLASSES = ["tomato_early_blight", "tomato_healthy", "tomato_late_blight", "tomato_leaf_spot", "tomato_other_problem"]
EB, HEALTHY, LATE, SPOT, OTHER = range(5)


def probs_row(top: int, p: float) -> list[float]:
    rest = (1.0 - p) / 4
    return [p if i == top else rest for i in range(5)]


def test_wilson_interval_known_value():
    low, high = wilson(10, 100)
    assert low == pytest.approx(0.0552, abs=1e-3) and high == pytest.approx(0.1744, abs=1e-3)


def test_wilson_zero_count_keeps_a_nonzero_upper_bound():
    low, high = wilson(0, 34)
    assert low == 0.0 and 0.05 < high < 0.15        # 0/34 must not read as "certainly zero"


def test_pass_probability_is_high_when_dev_is_far_below_the_limit():
    assert pass_probability(k=2, n=200, limit=0.10, future_n=100) > 0.99


def test_pass_probability_is_low_when_dev_is_above_the_limit():
    assert pass_probability(k=30, n=100, limit=0.10, future_n=100) < 0.01


def test_pass_probability_grows_with_a_looser_limit():
    a = pass_probability(k=15, n=116, limit=0.10, future_n=107)
    b = pass_probability(k=15, n=116, limit=0.20, future_n=107)
    assert 0.0 < a < b <= 1.0


def test_sheet_row_counts_each_outcome():
    known = np.array([probs_row(EB, 0.9), probs_row(EB, 0.5), probs_row(HEALTHY, 0.9),
                      probs_row(OTHER, 0.9), probs_row(LATE, 0.9)])
    known_y = np.array([EB, EB, LATE, SPOT, LATE])
    ood = np.array([probs_row(HEALTHY, 0.9), probs_row(OTHER, 0.9), probs_row(LATE, 0.5), probs_row(SPOT, 0.8)])
    row = sheet_row({"known": (known, known_y), "seen": ood, "novel": ood[:2]}, CLASSES, 0.7)

    # Accepted: rows 0, 2, 4 (row 1 under threshold, row 3 = 'other').
    assert row["coverage"]["k"] == 3 and row["coverage"]["n"] == 5
    assert row["accepted_error"]["k"] == 1 and row["accepted_error"]["n"] == 3      # row 2: late called healthy
    assert row["false_accept_seen"]["k"] == 2 and row["false_accept_seen"]["n"] == 4
    assert row["false_accept_novel"]["k"] == 1 and row["false_accept_novel"]["n"] == 2
    # The dangerous mistake: a sick plant told it looks healthy.
    assert row["sick_called_healthy"]["k"] == 1 and row["sick_called_healthy"]["n"] == 5
    assert row["ood_called_healthy"]["k"] == 1 and row["ood_called_healthy"]["n"] == 4


def test_sheet_row_handles_nothing_accepted():
    known = np.array([probs_row(EB, 0.3)])
    row = sheet_row({"known": (known, np.array([EB])), "seen": known, "novel": known}, CLASSES, 0.9)
    assert row["coverage"]["k"] == 0 and row["accepted_error"]["n"] == 0 and row["accepted_error"]["rate"] is None


def _row() -> dict:
    known = np.array([probs_row(EB, 0.9)] * 80 + [probs_row(OTHER, 0.9)] * 20)
    y = np.array([EB] * 100)
    ood = np.array([probs_row(OTHER, 0.9)] * 100)
    return sheet_row({"known": (known, y), "seen": ood, "novel": ood}, CLASSES, 0.7)


def test_sheet_row_produces_every_metric_the_gates_may_name():
    assert set(METRICS) <= set(_row()) and OTHER_CLASS == CLASSES[OTHER]


def test_every_blocking_gate_in_the_real_gates_file_gets_a_pass_probability():
    limits = load_limits()
    assert "sick_called_healthy" in limits["blocking"]
    p = add_pass_probabilities(_row(), limits)["p_pass"]
    assert set(p["blocking"]) == set(limits["blocking"])
    assert p["all_blocking_approx"] == pytest.approx(np.prod(list(p["blocking"].values())))


def test_every_metric_named_in_the_real_gates_file_exists():
    limits = load_limits()
    assert {*limits["blocking"], *limits["report_only"]} <= set(METRICS)


def test_limit_rounding_does_not_lose_an_image_to_floating_point():
    # 0.29 * 100 == 28.999999999999996; the limit is 29 images, so 29 of 100 must count as inside it.
    exact = pass_probability(29, 100, 0.29, 100)
    assert exact == pytest.approx(pass_probability(29, 100, 0.2900001, 100))
