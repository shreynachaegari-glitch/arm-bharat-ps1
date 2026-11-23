import numpy as np

from ps1.data.build_splits import OTHER_CLASS
from ps1.vision.abstain_oe import decide, evaluate

CLASSES = ["a", "b", OTHER_CLASS]


def test_other_prediction_is_never_accepted():
    probs = np.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05], [0.45, 0.1, 0.45]])
    accepted, pred = decide(probs, other_idx=2, threshold=0.0)
    assert accepted.tolist() == [False, True, True]
    accepted, _ = decide(probs, other_idx=2, threshold=0.5)
    assert accepted.tolist() == [False, True, False]          # low-confidence disease is abstained too


def test_evaluate_counts_other_on_known_as_error_and_reports_false_accepts():
    big = 10.0
    known_logits = np.array([[big, 0, 0], [0, big, 0], [0, 0, big]])     # third known image -> 'other'
    known_y = np.array([0, 1, 1])
    ood_logits = np.array([[0, 0, big], [big, 0, 0]])                   # one OOD image slips through
    sets = {"known_field": (known_logits, known_y),
            "ood_field_seen_types": (ood_logits, np.array([2, 2])),
            "ood_novel_india": (ood_logits, np.array([2, 2]))}
    r = evaluate(sets, CLASSES, temperature=1.0)
    assert abs(r["known_sent_to_other"] - 1 / 3) < 1e-9
    p0 = r["points"][0]                                                  # threshold 0
    assert abs(p0["known_coverage"] - 2 / 3) < 1e-9 and p0["accepted_error"] == 0.0
    assert p0["false_accept_ood_field_seen_types"] == 0.5
