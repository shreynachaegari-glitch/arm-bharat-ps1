"""Apply the pre-registered v9 fine-tuning rule (PS1 Fine-tune v9 Plan, written 22 Nov before
any v9 result). Two steps:

  python scripts/v9_summary.py screen    stage 1: seed-0 variants vs v8 seed 0 -> runs/v9_screen.json, last line
                                         "WINNER <variant>" or "WINNER none"
  python scripts/v9_summary.py confirm   stage 2: 3-seed means + NEU-test + India check -> runs/v9_summary.json

Score S = 1/2 PlantDoc out-of-fold known-field macro-F1 + 1/2 NEU-dev forced-choice macro-F1; everything at t = 0.80.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
THRESHOLD = 0.8
VARIANTS = {"v9a": "--epochs 20 --ema-start-epoch 8", "v9b": "--backbone-lr-mult 0.3",
            "v9c": "--backbone-lr-mult 1.0", "v9d": "--extra-oversample 2"}
BASELINE = "v8"
GUARD = {"pd_error": 0.03, "pd_fa_seen": 0.03, "neu_error": 0.03, "neu_sick_healthy": 0.01, "neu_fa_ood": 0.03}
SCREEN_MARGIN, CONFIRM_MARGIN = 0.02, 0.015
TEST_GUARD = {"error": 0.03, "sick_healthy": 0.01, "fa_ood": 0.03}


def worst(x: float | None) -> float:
    return 1.0 if x is None else x              # no answers at all: treat an error rate as the worst case


def guards_ok(base: dict, cand: dict) -> dict:
    return {k: worst(cand[k]) <= worst(base[k]) + m + 1e-12 for k, m in GUARD.items()}


def pick_winner(base: dict, cands: dict[str, dict]) -> tuple[str | None, dict]:
    checks = {}
    for name, c in cands.items():
        checks[name] = {**guards_ok(base, c), "margin": c["S"] >= base["S"] + SCREEN_MARGIN - 1e-12}
    passing = [n for n, ch in checks.items() if all(ch.values())]
    return (max(passing, key=lambda n: cands[n]["S"]) if passing else None), checks


def confirm_rule(v8: dict, v9: dict, test8: dict, test9: dict, india8: dict, india9: dict) -> dict:
    out = {"seed_mean_gain": v9["S"] >= v8["S"] + CONFIRM_MARGIN - 1e-12,
           **{f"seed_mean_{k}": ok for k, ok in guards_ok(v8, v9).items()},
           "test_f1": test9["f1_forced"] >= test8["f1_forced"] - 1e-12,
           **{f"test_{k}": worst(test9[k]) <= worst(test8[k]) + m + 1e-12 for k, m in TEST_GUARD.items()},
           "india_healthy": india9["unsupported_called_healthy"] <= india8["unsupported_called_healthy"],
           "india_wrong_disease": india9["wrong_disease"] <= india8["wrong_disease"]}
    out["adopt_v9"] = all(out.values())
    return out


def at_t(points: list[dict]) -> dict:
    return next(p for p in points if abs(p["threshold"] - THRESHOLD) < 1e-9)


def load(prefix: str, seed: int) -> dict:
    d = RUNS / f"{prefix}s{seed}_ema"
    oe, neu = json.loads((d / "abstention_oe_dev.json").read_text()), json.loads((d / "neu_dev.json").read_text())
    assert abs(neu["threshold"] - THRESHOLD) < 1e-9 and abs(neu["temperature"] - oe["temperature"]) < 1e-9, d
    p, n = at_t(oe["points"]), neu["pooled"]
    return {"pd_f1": oe["known_field_macro_f1"], "neu_f1": n["f1_forced"],
            "S": 0.5 * oe["known_field_macro_f1"] + 0.5 * n["f1_forced"],
            "pd_error": p["accepted_error"], "pd_fa_seen": p["false_accept_ood_field_seen_types"],
            "pd_coverage": p["known_coverage"], "neu_coverage": n["coverage"], "neu_error": n["accepted_error"],
            "neu_sick_healthy": n["sick_called_healthy"], "neu_fa_ood": n["false_accept_ood"]}


def mean_of(prefix: str) -> dict:
    per_seed = [load(prefix, s) for s in (0, 1, 2)]
    return {k: float(np.mean([worst(x[k]) if k in GUARD else x[k] for x in per_seed])) for k in per_seed[0]} | \
        {"S_values": [x["S"] for x in per_seed]}


def test_part(run: str) -> dict:
    d = json.loads((RUNS / run / "extra_eval_test.json").read_text())
    dep = d["deployed"]
    assert abs(dep["threshold"] - THRESHOLD) < 1e-9 and d["part"] == "test", run
    return {"f1_forced": d["forced_choice_macro_f1_present_classes"], "coverage": dep["coverage"]["rate"],
            "error": dep["accepted_error"]["rate"], "sick_healthy": dep["sick_called_healthy"]["rate"],
            "fa_ood": dep["false_accept_ood"]["rate"]}


def india(run: str) -> dict:
    g = json.loads((RUNS / run / "india_check.json").read_text())["groups"]
    return {"unsupported_called_healthy": g["unsupported_unseen"]["outcomes"].get("tomato_healthy", 0),
            "wrong_disease": g["early_blight_unseen"]["wrong_disease_accepts"]}


def line(name: str, m: dict) -> str:
    return (f"{name:5} S {m['S']:.3f} | PlantDoc F1 {m['pd_f1']:.3f} err {worst(m['pd_error']):.3f} FA seen "
            f"{m['pd_fa_seen']:.3f} | NEU-dev F1 {m['neu_f1']:.3f} answers {m['neu_coverage']:.3f} err "
            f"{worst(m['neu_error']):.3f} sick->healthy {m['neu_sick_healthy']:.3f} FA {m['neu_fa_ood']:.3f}")


def screen() -> None:
    base = load(BASELINE, 0)
    cands = {v: load(v, 0) for v in VARIANTS}
    winner, checks = pick_winner(base, cands)
    print(line(BASELINE, base))
    for v, m in cands.items():
        print(line(v, m), "| fails:", [k for k, ok in checks[v].items() if not ok] or "none")
    (RUNS / "v9_screen.json").write_text(json.dumps(
        {"baseline": base, "variants": cands, "changes": VARIANTS, "checks": checks, "winner": winner}, indent=1))
    print(f"WINNER {winner or 'none'}")


def confirm() -> None:
    winner = json.loads((RUNS / "v9_screen.json").read_text())["winner"]
    assert winner, "no stage-1 winner"
    v8, v9 = mean_of(BASELINE), mean_of(winner)
    t8, t9, i8, i9 = test_part("v8_final"), test_part("v9_final"), india("v8_final"), india("v9_final")
    rule = confirm_rule(v8, v9, t8, t9, i8, i9)
    print(line("v8 x3", v8))
    print(line(f"{winner} x3", v9))
    for name, t in (("v8", t8), ("v9", t9)):
        print(f"{name} NEU-test: F1 {t['f1_forced']:.3f} answers {t['coverage']:.3f} err {worst(t['error']):.3f} "
              f"sick->healthy {t['sick_healthy']:.3f} FA {t['fa_ood']:.3f}")
    print(f"India: v8 {i8}  v9 {i9}")
    print("rule:", rule)
    (RUNS / "v9_summary.json").write_text(json.dumps(
        {"winner": winner, "change": VARIANTS[winner], "seed_means": {"v8": v8, "v9": v9},
         "neu_test": {"v8": t8, "v9": t9}, "india": {"v8": i8, "v9": i9}, "rule": rule}, indent=1))


if __name__ == "__main__":
    {"screen": screen, "confirm": confirm}[sys.argv[1]]()
