"""v8 vs v7: the pre-registered rule (PS1 Extra Field Data v8 Plan, written before results).

Adopt v8 only if
  NEU held-out (finals):  F1_forced(v8) >= F1_forced(v7) + 0.05, error@0.80(v8) <= v7 + 0.03, sick->healthy(v8) <= v7 + 0.02
  PlantDoc OOF (3 seeds): mean F1(v8) >= mean F1(v7) - 0.01, mean seen-type FA@0.80(v8) <= v7 + 0.03
  python scripts/v8_summary.py      # table + verdict, also written to runs/v8_summary.json
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
T = 0.8
MARGINS = {"neu_f1_gain": 0.05, "neu_error": 0.03, "neu_sick_healthy": 0.02, "oof_f1": 0.01, "oof_fa_seen": 0.03}
OUT = ROOT / "runs/v8_summary.json"


def oof(prefix: str) -> dict:
    """Mean/sd over seeds of PlantDoc out-of-fold numbers at t=0.80 for runs/<prefix>{seed}_ema."""
    rs = [json.loads((ROOT / f"runs/{prefix}{s}_ema/abstention_oe_dev.json").read_text()) for s in SEEDS]
    pts = [next(p for p in r["points"] if abs(p["threshold"] - T) < 1e-9) for r in rs]
    def st(xs: list[float]) -> dict:
        return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs), "values": xs}
    return {"macro_f1": st([r["known_field_macro_f1"] for r in rs]),
            **{k: st([p[k] for p in pts]) for k in ("known_coverage", "accepted_error",
                                                    "false_accept_ood_field_seen_types", "false_accept_ood_novel_india")}}


def neu(run: str) -> dict:
    d = json.loads((ROOT / f"runs/{run}/extra_eval.json").read_text())
    dep = d["deployed"]
    assert abs(dep["threshold"] - T) < 1e-9, f"{run} decision threshold {dep['threshold']} is not {T}"
    return {"f1_forced": d["forced_choice_macro_f1_present_classes"], "coverage": dep["coverage"]["rate"],
            "error": dep["accepted_error"]["rate"] or 0.0, "sick_healthy": dep["sick_called_healthy"]["rate"],
            "fa_ood": dep["false_accept_ood"]["rate"], "raw": dep}


def rule(v7o: dict, v8o: dict, v7n: dict, v8n: dict) -> dict:
    checks = {
        "neu_f1_gain": v8n["f1_forced"] >= v7n["f1_forced"] + MARGINS["neu_f1_gain"],
        "neu_error": v8n["error"] <= v7n["error"] + MARGINS["neu_error"],
        "neu_sick_healthy": v8n["sick_healthy"] <= v7n["sick_healthy"] + MARGINS["neu_sick_healthy"],
        "oof_f1": v8o["macro_f1"]["mean"] >= v7o["macro_f1"]["mean"] - MARGINS["oof_f1"],
        "oof_fa_seen": v8o["false_accept_ood_field_seen_types"]["mean"]
        <= v7o["false_accept_ood_field_seen_types"]["mean"] + MARGINS["oof_fa_seen"],
    }
    return {**checks, "adopt_v8": all(checks.values())}


def main() -> int:
    v7o, v8o, v7n, v8n = oof("v7s"), oof("v8s"), neu("v7_final"), neu("v8_final")
    out = {"oof_t0.8": {"v7": v7o, "v8": v8o}, "neu_t0.8": {"v7": v7n, "v8": v8n}, "rule": rule(v7o, v8o, v7n, v8n)}
    OUT.write_text(json.dumps(out, indent=1))
    for name, o, n in (("v7", v7o, v7n), ("v8", v8o, v8n)):
        print(f"{name} | PlantDoc OOF F1 {o['macro_f1']['mean']:.3f} ± {o['macro_f1']['sd']:.3f}; t=0.8 answers "
              f"{o['known_coverage']['mean']:.0%} error {o['accepted_error']['mean']:.0%} FA seen "
              f"{o['false_accept_ood_field_seen_types']['mean']:.1%} | NEU F1(forced) {n['f1_forced']:.3f}; t=0.8 answers "
              f"{n['coverage']:.0%} error {n['error']:.0%} sick->healthy {n['sick_healthy']:.1%} FA {n['fa_ood']:.0%}")
    print("rule:", out["rule"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
