"""22 Nov: the pre-registered v9 rule (PS1 Fine-tune v9 Plan) picks a winner or none."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v9_summary import confirm_rule, pick_winner

BASE = {"S": 0.60, "pd_error": 0.10, "pd_fa_seen": 0.05, "neu_error": 0.15, "neu_sick_healthy": 0.0, "neu_fa_ood": 0.01}


def v(**kw) -> dict:
    return {**BASE, **kw}


def test_highest_score_passing_guardrails_and_margin_wins():
    cands = {"a": v(S=0.63), "b": v(S=0.66), "c": v(S=0.70, neu_error=0.20)}   # c breaks a guardrail (+0.05)
    winner, checks = pick_winner(BASE, cands)
    assert winner == "b" and checks["c"]["neu_error"] is False


def test_no_winner_when_gain_is_below_the_margin():
    winner, _ = pick_winner(BASE, {"a": v(S=0.615), "b": v(S=0.55)})
    assert winner is None


def test_sick_called_healthy_guardrail_is_tight():
    winner, checks = pick_winner(BASE, {"a": v(S=0.70, neu_sick_healthy=0.02)})
    assert winner is None and checks["a"]["neu_sick_healthy"] is False


def test_confirm_needs_seed_mean_gain_and_test_and_india():
    v8 = {"S": 0.60, **{k: BASE[k] for k in BASE if k != "S"}}
    v9 = {**v8, "S": 0.62}
    test8 = {"f1_forced": 0.55, "error": 0.13, "sick_healthy": 0.0, "fa_ood": 0.0}
    india8 = {"unsupported_called_healthy": 1, "wrong_disease": 0}
    ok = confirm_rule(v8, v9, test8, {**test8, "f1_forced": 0.56}, india8, india8)
    assert ok["adopt_v9"] is True
    worse_test = confirm_rule(v8, v9, test8, {**test8, "f1_forced": 0.54}, india8, india8)
    assert worse_test["adopt_v9"] is False and worse_test["test_f1"] is False
    worse_india = confirm_rule(v8, v9, test8, test8, india8, {**india8, "unsupported_called_healthy": 2})
    assert worse_india["adopt_v9"] is False
    small = confirm_rule(v8, {**v9, "S": 0.61}, test8, test8, india8, india8)
    assert small["seed_mean_gain"] is False
