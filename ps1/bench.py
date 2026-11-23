"""Latency / memory / thermal harness for the pipeline (G2, Experiment D).

Statistics follow Arm's ExecuTorch course `course_support/benchmarking.py` (warm-ups, per-run
perf_counter, mean / median / p50 / p90 / p95 / p99; BSD-3-Clause-Clear, ArmDeveloperEcosystem/
courses-code-executorch), extended per pipeline stage with: model-load (cold) time, peak RSS, the
system load average (a run on a busy CPU is flagged, not reported as a result), and on a Raspberry
Pi `vcgencmd measure_temp` / `get_throttled` before and after.
Usage: python -m ps1.bench --matrix scripts/g2/bench_matrix.json --images <dir-or-files> --out runs/bench/<name>
"""
import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ps1.contracts import Request
from ps1.llm.explain import LlamaConfig
from ps1.pipeline import build_config, default_llama_config, run

RUNNER_FAILED = {"error", "timeout"}   # explain() checker_result values where llama_main produced no usable output
BUSY_LOAD_PER_CORE = 0.5          # 1-minute load average per core above this marks the run "contaminated"


def stats(values_ms: list[float]) -> dict[str, float]:
    v = np.asarray(values_ms, dtype=np.float64)
    return {"count": int(v.size), "mean_ms": float(v.mean()), "median_ms": float(np.median(v)),
            "std_ms": float(v.std(ddof=0)), "min_ms": float(v.min()), "max_ms": float(v.max()),
            **{f"p{q}_ms": float(np.percentile(v, q)) for q in (50, 90, 95, 99)}}


def pi_thermal() -> dict[str, str] | None:
    """Raspberry Pi only: SoC temperature and throttling flags (None elsewhere)."""
    if not shutil.which("vcgencmd"):
        return None
    out = {}
    for key, cmd in (("temp", "measure_temp"), ("throttled", "get_throttled")):
        res = subprocess.run(["vcgencmd", cmd], capture_output=True, text=True, timeout=5, check=False)
        out[key] = res.stdout.strip()
    return out


def machine() -> dict:
    load1 = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cores": cores, "load1": load1, "busy": load1 / cores > BUSY_LOAD_PER_CORE}


def llama_outcomes(checker_results: list[str]) -> dict:
    """A runner that errored or timed out produced no generation, so its stage time says nothing about Llama."""
    failures = sum(r in RUNNER_FAILED for r in checker_results)
    return {"counts": dict(Counter(checker_results)), "runner_failures": failures,
            "valid": bool(checker_results) and failures == 0}


def missing_files(cfg_spec: dict) -> list[str]:
    """Artifacts a configuration needs that are not on this machine (one absent model must not stop the matrix)."""
    return [cfg_spec[k] for k in ("pte", "decision") if not Path(cfg_spec[k]).is_file()]


def bench_config(cfg_spec: dict, images: list[str], warmups: int, runs: int, root: Path) -> dict:
    llama = None
    if cfg_spec.get("llama"):
        base = default_llama_config(root)
        llama = LlamaConfig(command=base.command, model_path=base.model_path, tokenizer_path=base.tokenizer_path,
                            cpu_threads=cfg_spec.get("llama_threads", 4), timeout_s=cfg_spec.get("llama_timeout_s", 60))
    before = machine()
    t0 = time.perf_counter()
    cfg = build_config(cfg_spec["pte"], cfg_spec["decision"], allow_drafts=True, llama=llama)
    load_ms = (time.perf_counter() - t0) * 1000.0
    thermal_before = pi_thermal()
    for i in range(warmups):
        run(Request(crop="tomato", image_path=images[i % len(images)]), cfg)
    per_stage: dict[str, list[float]] = {}
    checker_results: list[str] = []
    for i in range(runs):
        res = run(Request(crop="tomato", image_path=images[i % len(images)]), cfg)
        for stage, ms in res.timings_ms.items():
            per_stage.setdefault(stage, []).append(ms)
        if res.screen2 is not None:
            checker_results.append(res.screen2.checker_result)
    after = machine()
    return {"config": cfg_spec, "warmups": warmups, "runs": runs, "model_load_ms": load_ms,
            "stages": {k: stats(v) for k, v in per_stage.items()},
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "thermal_before": thermal_before, "thermal_after": pi_thermal(),
            "machine_before": before, "machine_after": after,
            "contaminated": before["busy"] or after["busy"],
            "llama": llama_outcomes(checker_results) if llama else None,
            "per_run_ms": per_stage}


def image_list(spec: list[str]) -> list[str]:
    out: list[str] = []
    for item in spec:
        p = Path(item)
        out += sorted(str(x) for x in p.rglob("*") if x.suffix.lower() in {".jpg", ".jpeg", ".png"}) if p.is_dir() else [item]
    if not out:
        raise SystemExit("no images found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="JSON list of configurations (see scripts/g2/bench_matrix.json)")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list of config names to run")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = image_list(args.images)
    wanted = set(args.only.split(",")) - {""}
    for spec in json.loads(Path(args.matrix).read_text())["configs"]:
        if wanted and spec["name"] not in wanted:
            continue
        absent = missing_files(spec)
        if absent:
            print(f"{spec['name']:28} SKIPPED, missing: {', '.join(absent)}")
            continue
        runs = spec.get("runs", args.runs)
        result = bench_config(spec, images, spec.get("warmups", args.warmups), runs, root)
        (out / f"{spec['name']}.json").write_text(json.dumps(result, indent=1))
        s = result["stages"]["screen1_total"]
        flag = "  ** CONTAMINATED (busy CPU) **" if result["contaminated"] else ""
        llama = result["stages"].get("llama_explanation")
        if result["llama"] and not result["llama"]["valid"]:
            flag += f"  ** LLAMA INVALID, not a latency result: {result['llama']['counts']} **"
        print(f"{spec['name']:28} load {result['model_load_ms']:7.1f} ms | screen1 p50 {s['p50_ms']:7.1f} p95 {s['p95_ms']:7.1f} ms"
              + (f" | llama p50 {llama['p50_ms']:7.0f} ms" if llama else "") + f" | RSS {result['peak_rss_mb']:.0f} MB{flag}")


if __name__ == "__main__":
    main()
