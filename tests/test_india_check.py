from ps1.vision.india_check import outcome, summarise


def test_early_blight_counts_correct_and_wrong_accepts():
    s = summarise("early_blight_unseen", ["tomato_early_blight", "tomato_leaf_spot", outcome(None, "low_confidence")])
    assert (s["n"], s["accepted"], s["correct_accepts"], s["wrong_disease_accepts"]) == (3, 2, 1, 1)


def test_any_accept_on_unsupported_is_false_accept():
    s = summarise("unsupported_unseen", ["tomato_early_blight", outcome(None, "unsupported_label")])
    assert (s["accepted"], s["false_accepts"]) == (1, 1)
