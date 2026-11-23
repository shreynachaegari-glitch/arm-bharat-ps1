"""Indian-photo check of a deployed model on the Maharashtra photos it never trained on.

Runs the exact app path (sidecar-checked .pte -> app preprocessing -> logits -> bound decision config) on
- the 20 Maharashtra early-blight photos (ext_train; left out of training with --exclude-sources maharashtra)
- the 34 Maharashtra unsupported-problem photos (ood_dev; never used for training)
The 35 Maharashtra ood_test photos are sealed and never read here.
Usage: python -m ps1.vision.india_check --ckpt <last.pt> --pte <model>.pte --decision <decision>.json --out <dir>
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from ps1.decision import DecisionConfig, decide
from ps1.vision.dataset import read_manifest
from ps1.vision.runtime import load_vision

SOURCE = "maharashtra"
GROUPS = {"early_blight_unseen": ("ext_train", "tomato_early_blight"), "unsupported_unseen": ("ood_dev", None)}


def outcome(label: str | None, reason: str | None) -> str:
    return label if label is not None else f"abstain:{reason}"


def summarise(group: str, outcomes: list[str]) -> dict:
    """Counts per outcome plus the one number that matters for the group."""
    n, counts = len(outcomes), Counter(outcomes)
    accepted = sum(v for k, v in counts.items() if not k.startswith("abstain:"))
    summary = {"n": n, "outcomes": dict(counts.most_common()), "accepted": accepted}
    if group == "early_blight_unseen":
        summary["correct_accepts"] = counts["tomato_early_blight"]
        summary["wrong_disease_accepts"] = accepted - counts["tomato_early_blight"]
    else:
        summary["false_accepts"] = accepted     # any accepted label on an unsupported problem is wrong
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint the .pte came from (to prove Maharashtra was excluded)")
    ap.add_argument("--pte", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_args = json.loads(torch.load(args.ckpt, map_location="cpu", weights_only=True).get("args", "{}"))
    if SOURCE not in train_args.get("exclude_sources", "").split(","):
        raise SystemExit(f"{args.ckpt} trained on {SOURCE} images; this check needs a model that never saw them")

    vision, cfg = load_vision(args.pte), DecisionConfig.load(args.decision)
    rows = [r for r in read_manifest(taxonomy=vision.taxonomy) if r["source"] == SOURCE]
    results = {"pte_sha256": vision.pte_sha256, "decision": cfg.version, "groups": {}}
    for group, (split, canonical) in GROUPS.items():
        part = [r for r in rows if r["split"] == split and (canonical is None or r["canonical"] == canonical)]
        outcomes = []
        for r in part:
            d = decide(vision.logits(vision.preprocess(r["path"])), cfg, vision.classes, vision.pte_sha256)
            outcomes.append(outcome(d.label, d.abstain_reason.value if d.abstain_reason else None))
        results["groups"][group] = summarise(group, outcomes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "india_check.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
