"""Parse G1 llama_main logs into a per-run CSV and a summary.

Each run log holds the runner's `PyTorchObserver {json}` stats line and the
`/usr/bin/time -v` report. Times in the stats JSON are in milliseconds.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

OBSERVER = re.compile(r"PyTorchObserver (\{.*\})")
PEAK_RSS = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
EXIT_STATUS = re.compile(r"Exit status: (\d+)")
RUNTIME_LOG = re.compile(r"I \d\d:\d\d:\d\d\.\d+[^\n]*")


def parse_run(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    stats_match = OBSERVER.search(text)
    rss_match = PEAK_RSS.search(text)
    exit_match = EXIT_STATUS.search(text)
    row = {
        "run": log_path.stem,
        "ok": bool(stats_match) and (exit_match is None or exit_match.group(1) == "0"),
        "peak_rss_mb": round(int(rss_match.group(1)) / 1024, 1) if rss_match else None,
    }
    if not stats_match:
        return row
    s = json.loads(stats_match.group(1))
    decode_ms = s["inference_end_ms"] - s["first_token_ms"]
    decode_tokens = max(s["generated_tokens"] - 1, 0)
    row.update(
        prompt_tokens=s["prompt_tokens"],
        generated_tokens=s["generated_tokens"],
        load_s=round((s["model_load_end_ms"] - s["model_load_start_ms"]) / 1000, 3),
        ttft_s=round((s["first_token_ms"] - s["inference_start_ms"]) / 1000, 3),
        prefill_tok_s=round(s["prompt_tokens"] / max(s["prompt_eval_end_ms"] - s["inference_start_ms"], 1) * 1000, 1),
        decode_tok_s=round(decode_tokens / decode_ms * 1000, 1) if decode_ms > 0 else None,
        total_s=round((s["inference_end_ms"] - s["model_load_start_ms"]) / 1000, 3),
    )
    return row


def generated_text(log_path: Path) -> str:
    """Text the runner echoed between the prompt and the stats line (used to check greedy determinism)."""
    text = log_path.read_text(errors="replace")
    head = text.split("PyTorchObserver", 1)[0]
    body = head.split("<|start_header_id|>assistant<|end_header_id|>", 1)[-1]
    # stderr is merged into the run log: runtime lines ("I 00:00:02.99 executorch:...] RSS ...") interleave
    # with the tokens and carry per-run values, so they must go before comparing outputs.
    body = RUNTIME_LOG.sub("", body).replace("Reached to the end of generation", "").replace("<|eot_id|>", "")
    return " ".join(body.split())


def summarise(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return f"{key}: n/a"
    return f"{key}: median {statistics.median(values)}  min {min(values)}  max {max(values)}  (n={len(values)})"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    logs = sorted(out.glob("run_*.log"), key=lambda p: int(p.stem.split("_")[1]))
    if not logs:
        sys.exit(f"no run_*.log files in {out}")
    rows = [parse_run(p) for p in logs]

    fields = ["run", "ok", "prompt_tokens", "generated_tokens", "load_s", "ttft_s",
              "prefill_tok_s", "decode_tok_s", "total_s", "peak_rss_mb"]
    with open(out / "g1_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    texts = {generated_text(p) for p in logs}
    lines = [f"runs ok: {sum(r['ok'] for r in rows)}/{len(rows)}",
             f"distinct outputs (greedy should give 1): {len(texts)}"]
    lines += [summarise(rows, k) for k in ["load_s", "ttft_s", "prefill_tok_s", "decode_tok_s", "total_s", "peak_rss_mb"]]
    lines += ["", "output (run 1):", generated_text(logs[0])]
    summary = "\n".join(lines)
    (out / "g1_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
