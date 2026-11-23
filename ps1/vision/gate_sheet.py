"""Decision sheet for the promotion gates and the threshold, built BEFORE the one sealed run.

A dev point estimate does not say whether a gate will hold on a small sealed set. For each threshold this
reports counts, Wilson 95% intervals and P(sealed pass): the Jeffreys posterior from the dev counts, predicting
a new sample of the sealed set's size (beta-binomial). It also reports the mistake that matters most in the
field: a sick plant, or an unsupported problem, accepted as "healthy".

Dev only: out-of-fold logits from the fold models; sealed splits are never read (their sizes are constants).
Optimistic by construction: the threshold is picked on the same data it is scored on, and images are
treated as independent. Usage:
  python -m ps1.vision.gate_sheet --fold-ckpts runs/v5b_f0/last.pt runs/v5b_f1/last.pt \
      --abstention runs/v5b/abstention_oe_dev.json --out runs/v5b
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import betabinom

from ps1.data.build_splits import OTHER_CLASS
from ps1.vision.abstain_oe import collect, decide

HEALTHY_CLASS = "tomato_healthy"
THRESHOLDS = tuple(round(0.50 + 0.02 * i, 2) for i in range(21))     # 0.50 .. 0.90
SEALED_N = {"known": 259, "seen": 107, "novel": 35}                  # manifest counts; images not read
SEALED_SICK = 228                                                    # sealed known minus its 31 healthy (manifest count)
METRICS = ("coverage", "accepted_error", "false_accept_seen", "false_accept_novel", "sick_called_healthy",
           "ood_called_healthy")
JEFFREYS = 0.5
Z_95 = 1.959964
DEFAULT_GATES = Path(__file__).resolve().parents[2] / "scripts/final/gates.json"   # one source for the limits
ROUNDING_GUARD = 1e-9      # 0.29 * 100 is 28.999... in floats; the limit must not lose a whole image to that


def load_limits(path: str | Path = DEFAULT_GATES) -> dict:
    """Rule per metric from the gates file; the sheet can never drift from what the sealed run will enforce."""
    gates = json.loads(Path(path).read_text())
    return {"blocking": gates["blocking"], "report_only": gates.get("report_only", {})}


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95% interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + Z_95 ** 2 / n
    centre = (p + Z_95 ** 2 / (2 * n)) / denom
    half = Z_95 * math.sqrt(p * (1 - p) / n + Z_95 ** 2 / (4 * n ** 2)) / denom
    return (0.0 if k == 0 else centre - half), (1.0 if k == n else centre + half)


def pass_probability(k: int, n: int, limit: float, future_n: int) -> float:
    """P(rate on a new sample of future_n <= limit), given k events out of n on dev."""
    return float(betabinom(future_n, k + JEFFREYS, n - k + JEFFREYS).cdf(math.floor(limit * future_n + ROUNDING_GUARD)))


def _rate(k: int, n: int) -> dict:
    low, high = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else None), "low": low, "high": high}


def sheet_row(sets: dict, classes: list[str], threshold: float) -> dict:
    """`sets` holds calibrated probabilities: {"known": (probs, labels), "seen": probs, "novel": probs}."""
    other, healthy = classes.index(OTHER_CLASS), classes.index(HEALTHY_CLASS)
    probs, y = sets["known"]
    accepted, pred = decide(probs, other, threshold)
    sick = y != healthy
    seen_acc, seen_pred = decide(sets["seen"], other, threshold)
    novel_acc, _ = decide(sets["novel"], other, threshold)
    return {"threshold": threshold,
            "coverage": _rate(int(accepted.sum()), len(y)),
            "accepted_error": _rate(int((pred[accepted] != y[accepted]).sum()), int(accepted.sum())),
            "false_accept_seen": _rate(int(seen_acc.sum()), len(seen_acc)),
            "false_accept_novel": _rate(int(novel_acc.sum()), len(novel_acc)),
            "sick_called_healthy": _rate(int((accepted & sick & (pred == healthy)).sum()), int(sick.sum())),
            "ood_called_healthy": _rate(int((seen_acc & (seen_pred == healthy)).sum()), len(seen_acc))}


def add_pass_probabilities(row: dict, limits: dict, sealed: dict = SEALED_N, sealed_sick: int = SEALED_SICK) -> dict:
    """P(sealed pass) for every gate in the gates file. `all_blocking_approx` multiplies the blocking ones: a rough
    figure, because error, coverage and sick-called-healthy share the same known images and threshold."""
    expected_accepted = max(round((row["coverage"]["rate"] or 0.0) * sealed["known"]), 1)
    future = {"accepted_error": expected_accepted, "false_accept_seen": sealed["seen"], "false_accept_novel": sealed["novel"],
              "sick_called_healthy": sealed_sick, "ood_called_healthy": sealed["seen"], "coverage": sealed["known"]}

    def p_of(metric: str, rule: dict) -> float:
        r = row[metric]
        if "max" in rule:
            return pass_probability(r["k"], r["n"], rule["max"], future[metric])
        # rate >= c is the same event as "the rest" <= 1 - c.
        return pass_probability(r["n"] - r["k"], r["n"], 1 - rule["min"], future[metric])

    blocking = {m: p_of(m, rule) for m, rule in limits["blocking"].items()}
    report = {m: p_of(m, rule) for m, rule in limits["report_only"].items()}
    return {**row, "p_pass": {"blocking": blocking, "report_only": report,
                              "all_blocking_approx": float(np.prod(list(blocking.values())))}}


def _pct(r: dict) -> str:
    return "n/a" if r["rate"] is None else f"{r['rate']:.3f} ({r['k']}/{r['n']}; {r['low']:.2f}-{r['high']:.2f})"


def markdown(rows: list[dict], meta: dict) -> str:
    names = list(meta["limits"]["blocking"])
    head = (f"Temperature {meta['temperature']:.3f}; n known {meta['n']['known']}, seen-type OOD {meta['n']['seen']}, "
            f"novel Indian OOD {meta['n']['novel']}. Sealed sizes {SEALED_N}. Limits: {meta['limits']}.\n\n"
            "| t | coverage | accepted error | FA seen-type | FA novel India | sick → healthy | OOD → healthy | "
            + " | ".join(f"P({n})" for n in names) + " | P(all, rough) |\n|" + "---|" * (8 + len(names)) + "\n")
    body = ""
    for r in rows:
        p = r["p_pass"]
        probs = " | ".join(f"{p['blocking'][n]:.2f}" for n in names)
        body += (f"| {r['threshold']:.2f} | {_pct(r['coverage'])} | {_pct(r['accepted_error'])} | "
                 f"{_pct(r['false_accept_seen'])} | {_pct(r['false_accept_novel'])} | "
                 f"{_pct(r['sick_called_healthy'])} | {_pct(r['ood_called_healthy'])} | {probs} | "
                 f"{p['all_blocking_approx']:.2f} |\n")
    return head + body


def calibrated(logits: np.ndarray, temperature: float) -> np.ndarray:
    return torch.from_numpy(logits / temperature).softmax(1).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-ckpts", nargs="+", required=True)
    ap.add_argument("--abstention", required=True, help="abstention_oe_dev.json: the temperature the deployed config uses")
    ap.add_argument("--gates", default=str(DEFAULT_GATES), help="limits come from here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    limits = load_limits(args.gates)
    temperature = float(json.loads(Path(args.abstention).read_text())["temperature"])
    logits, classes, _ = collect(args.fold_ckpts)
    sets = {"known": (calibrated(logits["known_field"][0], temperature), logits["known_field"][1]),
            "seen": calibrated(logits["ood_field_seen_types"][0], temperature),
            "novel": calibrated(logits["ood_novel_india"][0], temperature)}
    rows = [add_pass_probabilities(sheet_row(sets, classes, t), limits) for t in THRESHOLDS]
    meta = {"temperature": temperature, "classes": classes, "limits": limits, "sealed_n": SEALED_N,
            "n": {"known": len(sets["known"][1]), "seen": len(sets["seen"]), "novel": len(sets["novel"])},
            "note": "dev estimate; threshold chosen and scored on the same out-of-fold data; PlantDoc-like field data"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_sheet.json").write_text(json.dumps({**meta, "rows": rows}, indent=1))
    text = markdown(rows, meta)
    (out / "gate_sheet.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
