"""G2 Experiment D cross-check: parse llama.cpp runs and compare them with the ExecuTorch G1 runs.

A speed and memory baseline only. The weights differ (GGUF Q4_K_M vs our SpinQuant INT4 .pte), so this says
nothing about output quality, and the shipped runtime stays ExecuTorch 0.6.0.
Usage: python -m ps1.llamacpp_baseline <llamacpp_log_dir> [--executorch logs/g1_<run>/g1_runs.csv]
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

NUM = r"([\d.]+)"
LOAD = re.compile(rf"common_perf_print:\s+load time =\s+{NUM} ms")
PROMPT = re.compile(rf"common_perf_print: prompt eval time =\s+{NUM} ms /\s+(\d+) tokens \(.*?{NUM} tokens per second")
DECODE = re.compile(rf"common_perf_print:\s+eval time =\s+{NUM} ms /\s+(\d+) runs\s+\(.*?{NUM} tokens per second")
PEAK_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
EXIT_STATUS = re.compile(r"Exit status: (\d+)")
METRICS = ("decode_tok_s", "prefill_tok_s", "load_s", "peak_rss_mb")


def parse_log(text: str) -> dict:
    """Missing timings stay None and mark the run not ok; a failed run must never read as a fast one."""
    load, prompt, decode = LOAD.search(text), PROMPT.search(text), DECODE.search(text)
    rss, status = PEAK_RSS.search(text), EXIT_STATUS.search(text)
    return {"ok": bool(load and prompt and decode) and (status is None or status.group(1) == "0"),
            "load_s": float(load.group(1)) / 1000 if load else None,
            "prompt_tokens": int(prompt.group(2)) if prompt else None,
            "prefill_tok_s": float(prompt.group(3)) if prompt else None,
            "decode_runs": int(decode.group(2)) if decode else None,
            "decode_tok_s": float(decode.group(3)) if decode else None,
            "peak_rss_mb": int(rss.group(1)) / 1024 if rss else None}


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    medians = {m: statistics.median(r[m] for r in ok) if ok else None for m in METRICS}
    return {"runs_ok": len(ok), "runs_total": len(rows), **medians}


def compare(executorch: dict, llamacpp: dict) -> dict:
    return {m: {"executorch": executorch[m], "llamacpp": llamacpp[m],
                "llamacpp_over_executorch": llamacpp[m] / executorch[m]} for m in METRICS}


def executorch_summary(csv_path: str) -> dict:
    """g1_runs.csv from scripts/g1/parse_g1.py, reduced to the same metrics."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ok"] == "True"]
    if not rows:
        raise SystemExit(f"no successful runs in {csv_path}")
    return {m: statistics.median(float(r[m]) for r in rows) for m in METRICS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--executorch", help="g1_runs.csv from the same machine, for the comparison")
    args = ap.parse_args()
    logs = sorted(Path(args.log_dir).glob("run_*.log"))
    if not logs:
        raise SystemExit(f"no run_*.log files in {args.log_dir}")
    result = {"llamacpp": summarise([parse_log(p.read_text(errors="replace")) for p in logs])}
    if args.executorch and result["llamacpp"]["runs_ok"]:
        result["comparison"] = compare(executorch_summary(args.executorch), result["llamacpp"])
    (Path(args.log_dir) / "llamacpp_summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
