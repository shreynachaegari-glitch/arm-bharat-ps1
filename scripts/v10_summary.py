"""Apply the pre-registered v10 rule (EfficientNet-B2 at 260 px; PS1 Bigger Backbone v10 Plan,
written 23 Nov before any v10 result). Scoring is the v9 one (scripts/v9_summary.py).

  python scripts/v10_summary.py seeds     stage A: 3-seed means vs v8 -> runs/v10_seeds.json, last line PASS|FAIL
  python scripts/v10_summary.py confirm   stage B: + NEU-test, India check, speed, parity -> runs/v10_summary.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v9_summary as v9

PREFIX, FINAL = "v10", "v10_final"
SPEED_MAX = 3.0                 # x v8_final's x86 .pte median (Sreyesh 23 Nov: up to ~3x B0)
PARITY_MIN_DELTA = -0.02        # .pte minus FP32 macro-F1 on PlantVillage val (Arm build-mode rule)


def seeds_rule(v8: dict, v10: dict) -> dict:
    out = {"seed_mean_gain": v10["S"] >= v8["S"] + v9.CONFIRM_MARGIN - 1e-12,
           **{f"seed_mean_{k}": ok for k, ok in v9.guards_ok(v8, v10).items()}}
    out["pass"] = all(out.values())
    return out


def final_rule(v8: dict, v10: dict, test8: dict, test10: dict, india8: dict, india10: dict,
               speed8: dict, speed10: dict) -> dict:
    out = {k: ok for k, ok in v9.confirm_rule(v8, v10, test8, test10, india8, india10).items() if k != "adopt_v9"}
    out["speed"] = speed10["ms"] <= SPEED_MAX * speed8["ms"] + 1e-12
    out["parity"] = speed10["parity_delta"] >= PARITY_MIN_DELTA - 1e-12
    out["adopt_v10"] = all(out.values())
    return out


def speed(run: str) -> dict:
    parity = json.loads(next((v9.RUNS / run / "export").glob("*_fp32_xnnpack_parity.json")).read_text())["pv_val"]
    return {"ms": parity["pte_median_ms_x86"], "parity_delta": parity["delta_pte_minus_fp32"]}


def seeds() -> None:
    v8, v10 = v9.mean_of(v9.BASELINE), v9.mean_of(PREFIX)
    rule = seeds_rule(v8, v10)
    print(v9.line("v8 x3", v8))
    print(v9.line("v10x3", v10))
    for s in (0, 1, 2):
        print(v9.line(f"v10s{s}", v9.load(PREFIX, s)))
    print("rule:", rule)
    (v9.RUNS / "v10_seeds.json").write_text(json.dumps({"seed_means": {"v8": v8, "v10": v10}, "rule": rule}, indent=1))
    print("PASS" if rule["pass"] else "FAIL")


def confirm() -> None:
    v8, v10 = v9.mean_of(v9.BASELINE), v9.mean_of(PREFIX)
    t8, t10 = v9.test_part("v8_final"), v9.test_part(FINAL)
    i8, i10 = v9.india("v8_final"), v9.india(FINAL)
    s8, s10 = speed("v8_final"), speed(FINAL)
    rule = final_rule(v8, v10, t8, t10, i8, i10, s8, s10)
    for name, t in (("v8", t8), ("v10", t10)):
        print(f"{name} NEU-test: F1 {t['f1_forced']:.3f} answers {t['coverage']:.3f} err {v9.worst(t['error']):.3f} "
              f"sick->healthy {t['sick_healthy']:.3f} FA {t['fa_ood']:.3f}")
    print(f"India: v8 {i8}  v10 {i10}")
    print(f"speed x86: v8 {s8}  v10 {s10}")
    print("rule:", rule)
    (v9.RUNS / "v10_summary.json").write_text(json.dumps(
        {"seed_means": {"v8": v8, "v10": v10}, "neu_test": {"v8": t8, "v10": t10}, "india": {"v8": i8, "v10": i10},
         "speed": {"v8": s8, "v10": s10}, "rule": rule}, indent=1))


if __name__ == "__main__":
    {"seeds": seeds, "confirm": confirm}[sys.argv[1]]()
