import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import v10_summary as v

BASE = {"S": 0.591, "pd_error": 0.10, "pd_fa_seen": 0.066, "neu_error": 0.25, "neu_sick_healthy": 0.0, "neu_fa_ood": 0.0}
TEST8 = {"f1_forced": 0.582, "error": 0.113, "sick_healthy": 0.0, "fa_ood": 0.0}
INDIA8 = {"unsupported_called_healthy": 1, "wrong_disease": 0}
SPEED8 = {"ms": 5.06, "parity_delta": 0.0}


def test_seed_stage_needs_the_margin_and_the_guardrails():
    assert v.seeds_rule(BASE, {**BASE, "S": 0.606})["pass"]
    assert not v.seeds_rule(BASE, {**BASE, "S": 0.605})["pass"]                      # margin is +0.015
    assert not v.seeds_rule(BASE, {**BASE, "S": 0.70, "neu_error": 0.29})["pass"]    # guardrail +0.03


def test_final_stage_adds_speed_and_parity_to_the_v9_checks():
    good = {"v8": BASE, "v10": {**BASE, "S": 0.62}, "test8": TEST8, "test10": TEST8, "india8": INDIA8, "india10": INDIA8,
                "speed8": SPEED8, "speed10": {"ms": 12.0, "parity_delta": 0.0}}
    assert v.final_rule(**good)["adopt_v10"]
    slow = v.final_rule(**{**good, "speed10": {"ms": 15.3, "parity_delta": 0.0}})
    assert not slow["speed"] and not slow["adopt_v10"]                               # > 3x v8's 5.06 ms
    drift = v.final_rule(**{**good, "speed10": {"ms": 9.0, "parity_delta": -0.03}})
    assert not drift["parity"] and not drift["adopt_v10"]
    worse = v.final_rule(**{**good, "test10": {**TEST8, "f1_forced": 0.58}})
    assert not worse["test_f1"] and not worse["adopt_v10"]
