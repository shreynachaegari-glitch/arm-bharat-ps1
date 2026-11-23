"""Golden-logits check: the same .pte and images must give the same answers in another environment.

Write once on the x86 dev box (PyPI executorch 0.6.0), then check on the Pi 5 or against a source-built wheel.
It separates three findings: logits drift (runtime/kernels), top-1 flips (user-visible), and preprocessing
tensors that differ (JPEG decoder or resize differences, which shift logits for a different reason).
Non-sealed images only. Usage:
  python -m ps1.vision.golden --write --pte <m>.pte --images runs/bench/images --golden runs/v6_final/golden.json
  python -m ps1.vision.golden --check --pte <m>.pte --golden runs/v6_final/golden.json
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from ps1.vision.runtime import load_vision

SEALED_LISTING = Path(__file__).resolve().parents[2] / "data/sealed_test.sha256"
DEFAULT_TOLERANCE = 0.01      # absolute logit difference; logits here are of order 1-10, so about 0.1%


def entry(image: str, logits: np.ndarray, tensor_sha256: str) -> dict:
    return {"image": image, "logits": [float(x) for x in logits], "top1": int(np.argmax(logits)),
            "tensor_sha256": tensor_sha256}


def compare(golden: list[dict], current: list[dict], tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Pass = same image set, every top-1 identical, every logit within tolerance."""
    want, got = {e["image"]: e for e in golden}, {e["image"]: e for e in current}
    shared = sorted(set(want) & set(got))
    diffs = {name: float(np.max(np.abs(np.subtract(want[name]["logits"], got[name]["logits"])))) for name in shared}
    worst = max(diffs, key=diffs.get) if diffs else None
    agreement = sum(want[n]["top1"] == got[n]["top1"] for n in shared) / len(shared) if shared else 0.0
    missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
    max_diff = diffs[worst] if worst else float("inf")
    return {"n": len(shared), "passed": not missing and not extra and agreement == 1.0 and max_diff <= tolerance,
            "top1_agreement": agreement, "max_abs_diff": max_diff, "worst_image": worst, "tolerance": tolerance,
            "tensor_mismatches": sum(want[n]["tensor_sha256"] != got[n]["tensor_sha256"] for n in shared),
            "missing": missing, "extra": extra}


def sealed_overlap(images: list[Path], listing: Path = SEALED_LISTING) -> list[str]:
    """Names of images whose bytes are in the sealed listing. The listing is absent on the Pi, where no sealed data goes."""
    if not listing.exists():
        return []
    sealed = {line.split(maxsplit=2)[1] for line in listing.read_text().splitlines() if line.strip()}
    return [p.name for p in images if hashlib.sha256(p.read_bytes()).hexdigest() in sealed]


def score(pte: str, images: list[Path]) -> tuple[str, list[dict]]:
    vision = load_vision(pte)
    out = []
    for path in images:
        batch = vision.preprocess(path)
        sha = hashlib.sha256(batch.numpy().tobytes()).hexdigest()
        out.append(entry(path.name, vision.logits(batch), sha))
    return vision.pte_sha256, out


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--pte", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--images", default="runs/bench/images", help="folder of non-sealed images")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--report", help="also save the check report as JSON here")
    args = ap.parse_args()

    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        sys.exit(f"no images in {args.images}")
    overlap = sealed_overlap(images)
    if overlap:
        sys.exit(f"sealed images must not be used here: {overlap[:5]}")
    sha, current = score(args.pte, images)
    if args.write:
        Path(args.golden).write_text(json.dumps({"pte_sha256": sha, "machine": platform.platform(),
                                                 "entries": current}, indent=1))
        print(f"wrote {len(current)} golden entries for pte {sha[:12]} to {args.golden}")
        return
    golden = json.loads(Path(args.golden).read_text())
    if golden["pte_sha256"] != sha:
        sys.exit(f"golden file is for pte {golden['pte_sha256'][:12]}, this is {sha[:12]}")
    report = {"machine": platform.platform(), **compare(golden["entries"], current, args.tolerance)}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
