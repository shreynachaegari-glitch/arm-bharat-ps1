import numpy as np

from ps1.bench import image_list, llama_outcomes, machine, stats


def test_stats_match_numpy_percentiles():
    v = list(np.arange(1, 101, dtype=float))
    s = stats(v)
    assert s["count"] == 100 and s["median_ms"] == 50.5
    assert abs(s["p95_ms"] - np.percentile(v, 95)) < 1e-9 and s["min_ms"] == 1 and s["max_ms"] == 100


def test_image_list_expands_folders(tmp_path):
    (tmp_path / "a").mkdir()
    for name in ("x.jpg", "y.PNG", "notes.txt"):
        (tmp_path / "a" / name).write_bytes(b"0")
    assert [p.split("/")[-1] for p in image_list([str(tmp_path / "a")])] == ["x.jpg", "y.PNG"]


def test_machine_reports_busy_flag():
    m = machine()
    assert {"cores", "load1", "busy"} <= set(m) and isinstance(m["busy"], bool)


def test_a_dead_llama_runner_is_flagged_not_reported_as_fast():
    # explain() falls back to the template on error, which is right for the farmer and wrong for a benchmark:
    # exit 127 took 2 ms and read as a superb Llama latency.
    s = llama_outcomes(["error", "error", "error"])
    assert s["valid"] is False and s["runner_failures"] == 3


def test_a_timeout_also_invalidates_the_llama_latency():
    assert llama_outcomes(["pass", "timeout"])["valid"] is False


def test_checker_rejection_is_a_real_run_and_keeps_the_latency_valid():
    # The model did generate; the checker then refused the text.
    s = llama_outcomes(["pass", "fail", "pass"])
    assert s["valid"] is True and s["runner_failures"] == 0 and s["counts"] == {"pass": 2, "fail": 1}


def test_no_llama_runs_at_all_is_not_valid():
    assert llama_outcomes([])["valid"] is False
