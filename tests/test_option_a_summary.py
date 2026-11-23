"""The pre-registered Option A rule picks EMA only when F1 is not worse and novel false accepts stay in margin."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("oas", Path(__file__).resolve().parents[1] / "scripts/option_a_summary.py")
oas = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oas)


def write(root: Path, arm: str, f1: float, fa78: float) -> None:
    for seed in (0, 1, 2):
        d = root / f"runs/v7s{seed}_{arm}"
        d.mkdir(parents=True)
        pts = [{"threshold": t, "known_coverage": 0.5, "accepted_error": 0.1, "false_accept_ood_field_seen_types": 0.1,
                "false_accept_ood_novel_india": fa78 if t == 0.78 else 0.0} for t in (0.7, 0.78)]
        (d / "abstention_oe_dev.json").write_text(json.dumps({"known_field_macro_f1": f1 + 0.001 * seed, "points": pts}))


@pytest.mark.parametrize("raw,ema,arm", [
    ((0.69, 0.05), (0.70, 0.06), "ema"),      # better F1, FA within +0.03
    ((0.69, 0.05), (0.68, 0.05), "raw"),      # worse F1
    ((0.69, 0.05), (0.72, 0.10), "raw"),      # better F1 but FA beyond the margin
])
def test_rule(tmp_path, monkeypatch, raw, ema, arm):
    write(tmp_path, "raw", *raw)
    write(tmp_path, "ema", *ema)
    monkeypatch.setattr(oas, "ROOT", tmp_path)
    assert oas.summarise()["rule"]["arm"] == arm
