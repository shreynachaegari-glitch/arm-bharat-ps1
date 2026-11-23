"""Option A summary: raw last.pt vs EMA over seeds 0-2, and the pre-registered decision rule.

Rule (PS1 Option A Seeds and EMA Plan, written before any result):
  adopt EMA iff  mean_F1(ema) >= mean_F1(raw)  AND  mean_FA_novel@0.78(ema) <= mean_FA_novel@0.78(raw) + 0.03.
  python scripts/option_a_summary.py              # table + verdict, also written to runs/option_a_summary.json
  python scripts/option_a_summary.py --print-arm  # prints only "ema" or "raw" (used by run_option_a.sh)
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
ARMS = ("raw", "ema")
FA_MARGIN = 0.03                      # one novel Indian photo in 34
OUT = ROOT / "runs/option_a_summary.json"


def point(result: dict, t: float) -> dict:
    return next(p for p in result["points"] if abs(p["threshold"] - t) < 1e-9)


def load(arm: str) -> list[dict]:
    results = []
    for seed in SEEDS:
        path = ROOT / f"runs/v7s{seed}_{arm}/abstention_oe_dev.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        results.append(json.loads(path.read_text()))
    return results


def stats(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "values": values}


def summarise() -> dict:
    out = {}
    for arm in ARMS:
        rs = load(arm)
        out[arm] = {"macro_f1": stats([r["known_field_macro_f1"] for r in rs])}
        for t in (0.7, 0.78):
            pts = [point(r, t) for r in rs]
            out[arm][f"t{t}"] = {k: stats([p[k] for p in pts]) for k in
                                 ("known_coverage", "accepted_error", "false_accept_ood_field_seen_types",
                                  "false_accept_ood_novel_india")}
    f1_ok = out["ema"]["macro_f1"]["mean"] >= out["raw"]["macro_f1"]["mean"]
    fa = "false_accept_ood_novel_india"
    fa_ok = out["ema"]["t0.78"][fa]["mean"] <= out["raw"]["t0.78"][fa]["mean"] + FA_MARGIN
    out["rule"] = {"f1_not_worse": f1_ok, "novel_fa_within_margin": fa_ok, "arm": "ema" if (f1_ok and fa_ok) else "raw"}
    return out


def fmt(s: dict, pct: bool = False) -> str:
    if pct:
        return f"{s['mean']:.0%} ± {s['sd']:.0%} ({s['min']:.0%}-{s['max']:.0%})"
    return f"{s['mean']:.3f} ± {s['sd']:.3f} ({s['min']:.3f}-{s['max']:.3f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-arm", action="store_true")
    args = ap.parse_args()
    out = summarise()
    if args.print_arm:
        print(out["rule"]["arm"])
        return 0
    OUT.write_text(json.dumps(out, indent=1))
    print("arm | macro-F1 (mean ± sd, min-max over 3 seeds) | t=0.7 answers | t=0.7 error | t=0.78 answers | "
          "t=0.78 error | t=0.78 FA novel Indian")
    for arm in ARMS:
        a = out[arm]
        print(f"{arm} | {fmt(a['macro_f1'])} | {fmt(a['t0.7']['known_coverage'], True)} | "
              f"{fmt(a['t0.7']['accepted_error'], True)} | {fmt(a['t0.78']['known_coverage'], True)} | "
              f"{fmt(a['t0.78']['accepted_error'], True)} | {fmt(a['t0.78']['false_accept_ood_novel_india'], True)}")
    print(f"rule: {out['rule']}  -> ship {out['rule']['arm']}")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
