"""The v8 rule, both ways, on made-up numbers."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("v8s", Path(__file__).resolve().parents[1] / "scripts/v8_summary.py")
v8s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v8s)


def o(f1: float, fa: float) -> dict:
    return {"macro_f1": {"mean": f1}, "false_accept_ood_field_seen_types": {"mean": fa}}


def n(f1: float, err: float, sick: float) -> dict:
    return {"f1_forced": f1, "error": err, "sick_healthy": sick}


def test_adopts_only_when_neu_gain_is_real_and_plantdoc_holds():
    assert v8s.rule(o(0.695, 0.05), o(0.69, 0.06), n(0.50, 0.20, 0.02), n(0.60, 0.21, 0.03))["adopt_v8"]


def test_small_neu_gain_or_plantdoc_drop_keeps_v7():
    assert not v8s.rule(o(0.695, 0.05), o(0.69, 0.06), n(0.50, 0.20, 0.02), n(0.54, 0.20, 0.02))["adopt_v8"]
    assert not v8s.rule(o(0.695, 0.05), o(0.67, 0.06), n(0.50, 0.20, 0.02), n(0.70, 0.10, 0.0))["adopt_v8"]
    assert not v8s.rule(o(0.695, 0.05), o(0.70, 0.05), n(0.50, 0.20, 0.02), n(0.70, 0.10, 0.05))["adopt_v8"]
