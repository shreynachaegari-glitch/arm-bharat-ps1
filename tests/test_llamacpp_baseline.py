from pathlib import Path

import pytest

from ps1.llamacpp_baseline import compare, parse_log, summarise

# Real output of llama-completion at the pinned commit cd959ba (stories260K tiny model, x86 container).
REAL_LOG = (Path(__file__).parent / "fixtures/llamacpp_cd959ba_stories260k.log").read_text()


def test_parses_the_real_log_of_the_pinned_commit():
    row = parse_log(REAL_LOG)
    assert row["ok"] and row["load_s"] == pytest.approx(0.00238)
    assert row["prompt_tokens"] == 5 and row["prefill_tok_s"] == pytest.approx(16722.41)
    assert row["decode_runs"] == 39 and row["decode_tok_s"] == pytest.approx(8873.72)
    assert row["peak_rss_mb"] == pytest.approx(14988 / 1024, abs=0.1)


def test_a_log_without_timings_is_not_ok_rather_than_zero():
    row = parse_log("error: failed to load model\n\tExit status: 1\n")
    assert not row["ok"] and row["decode_tok_s"] is None


def test_nonzero_exit_is_not_ok_even_with_timings():
    assert not parse_log(REAL_LOG.replace("Exit status: 0", "Exit status: 137"))["ok"]


def test_summary_uses_medians_of_ok_runs_only():
    rows = [{"ok": True, "decode_tok_s": v, "prefill_tok_s": 1.0, "load_s": 1.0, "peak_rss_mb": 1.0} for v in (10.0, 20.0, 30.0)]
    rows.append({"ok": False, "decode_tok_s": None, "prefill_tok_s": None, "load_s": None, "peak_rss_mb": None})
    s = summarise(rows)
    assert s["runs_ok"] == 3 and s["runs_total"] == 4 and s["decode_tok_s"] == 20.0


def test_compare_reports_the_ratio_against_executorch():
    table = compare({"decode_tok_s": 50.0, "prefill_tok_s": 100.0, "load_s": 2.0, "peak_rss_mb": 1000.0},
                    {"decode_tok_s": 25.0, "prefill_tok_s": 100.0, "load_s": 1.0, "peak_rss_mb": 500.0})
    assert table["decode_tok_s"] == {"executorch": 50.0, "llamacpp": 25.0, "llamacpp_over_executorch": 0.5}
