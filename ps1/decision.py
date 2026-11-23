"""Decision engine: logits -> accepted label or a reasoned abstention (v4 2.5, 2.7; outlier exposure).

The config (temperature, threshold, class order) is bound to one .pte by its sha256; a mismatch is an
error, so thresholds tuned for one model can never be applied to another.
Build one: python -m ps1.decision --abstention runs/v4/abstention_oe_dev.json \\
             --sidecar runs/v4_f0/export/<model>.json --threshold 0.7 --out runs/v4_f0/export/decision.json
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ps1.contracts import AbstainReason
from ps1.data.build_splits import OTHER_CLASS


@dataclass(frozen=True)
class DecisionConfig:
    version: str
    pte_sha256: str
    classes: tuple[str, ...]
    temperature: float
    threshold: float
    source: str                       # where T and the threshold came from (dev estimate until the sealed run)

    @classmethod
    def load(cls, path: str | Path) -> "DecisionConfig":
        d = json.loads(Path(path).read_text())
        return cls(**{**d, "classes": tuple(d["classes"])})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))


@dataclass(frozen=True)
class Decision:
    label: str | None                 # None when abstained
    abstain_reason: AbstainReason | None
    score: float                      # calibrated top probability; internal only, never shown as a probability


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def decide(logits: np.ndarray, cfg: DecisionConfig, model_classes: tuple[str, ...], model_sha: str) -> Decision:
    if model_sha != cfg.pte_sha256 or tuple(model_classes) != cfg.classes:
        raise ValueError("decision config is bound to a different model (sha or class order differ)")
    probs = softmax(np.asarray(logits, dtype=np.float64) / cfg.temperature)
    top = int(probs.argmax())
    label, score = cfg.classes[top], float(probs[top])
    if label == OTHER_CLASS:
        return Decision(None, AbstainReason.UNSUPPORTED_LABEL, score)
    if score < cfg.threshold:
        return Decision(None, AbstainReason.LOW_CONFIDENCE, score)
    return Decision(label, None, score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstention", required=True, help="abstention_oe_dev.json (temperature source)")
    ap.add_argument("--sidecar", required=True, help="the .pte sidecar JSON from ps1.vision.export")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ab, side = json.loads(Path(args.abstention).read_text()), json.loads(Path(args.sidecar).read_text())
    if ab["classes"] != side["classes"]:
        raise SystemExit("abstention file and model disagree on class order")
    cfg = DecisionConfig(version=f"d-{side['pte_sha256'][:8]}-t{args.threshold}", pte_sha256=side["pte_sha256"],
                         classes=tuple(side["classes"]), temperature=float(ab["temperature"]),
                         threshold=args.threshold, source=f"dev estimate from {args.abstention}")
    cfg.save(args.out)
    print(json.dumps(asdict(cfg), indent=1))


if __name__ == "__main__":
    main()
