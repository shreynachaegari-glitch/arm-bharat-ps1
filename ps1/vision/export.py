"""W5: INT8 post-training quantization + XNNPACK .pte export + FP32-vs-INT8 parity (v4 2.4).

Normalisation is baked into the exported graph: the .pte takes a float RGB tensor in [0, 1]
(resize 256, centre-crop 224), so the app cannot get mean/std wrong. A JSON sidecar records
class order, preprocessing, hashes and runtime, so outputs cannot be mislabelled.

INT8 calibration = pv_calib + the field half the checkpoint trained on (never sealed). Parity compares
FP32 eager with the .pte through the ExecuTorch runtime on pv_val and the field half the checkpoint
never saw: macro-F1, per-class recall, top-1 agreement and max logit difference.
FP32 is the default (INT8 failed the parity gate on v3); --precision int8 keeps the PTQ path.
Usage: python -m ps1.vision.export --ckpt runs/v4_f0/last.pt --out runs/v4_f0/export --precision fp32
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import executorch
import numpy as np
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from torch.ao.quantization import (
    allow_exported_model_train_eval,
    move_exported_model_to_eval,
)
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ps1.vision.checkpoint import ckpt_meta, load_fp32
from ps1.vision.dataset import (
    IMAGE_SIZE,
    MEAN,
    STD,
    ManifestDataset,
    eval_transform,
    input_size,
    read_manifest,
    resize_for,
    select,
)
from ps1.vision.train import predict, report, split_field

CALIBRATION_IMAGES = 512


class Normalized(torch.nn.Module):
    """Deployed graph: [0, 1] RGB in, logits out; ImageNet normalisation inside the .pte."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def calibration_rows(lab: list[dict], field: list[dict], n: int = CALIBRATION_IMAGES, seed: int = 0) -> list[dict]:
    """Seeded, class-balanced calibration sample: about half field rows, the rest lab rows, shuffled.

    Taking the first N manifest rows would be class-ordered, and the quantize loop stops after N images.
    """
    rng = random.Random(seed)
    field_part = rng.sample(field, min(len(field), n // 2))
    by_class: dict[str, list[dict]] = {}
    for r in lab:
        by_class.setdefault(r["canonical"], []).append(r)
    per_class = max((n - len(field_part)) // max(len(by_class), 1), 1)
    lab_part = [r for rs in by_class.values() for r in rng.sample(rs, min(len(rs), per_class))]
    rows = field_part + lab_part
    rng.shuffle(rows)
    return rows[:n]


def deploy_transform(size: int = IMAGE_SIZE) -> T.Compose:
    """What the app must do before the .pte: identical to eval_transform minus Normalize."""
    return T.Compose([T.Resize(resize_for(size)), T.CenterCrop(size), T.ToTensor()])


def sidecar_input(size: int) -> dict:
    """The sidecar's input contract; ps1.vision.runtime.input_spec reads it back."""
    return {"layout": "NCHW", "shape": [1, 3, size, size], "dtype": "float32", "range": [0, 1],
            "resize": resize_for(size),
            "preprocess": f"EXIF-transpose; RGB; resize shorter side to {resize_for(size)} (bilinear); "
                          f"centre-crop {size}; /255",
            "normalisation": "inside the .pte (ImageNet mean/std)"}


def quantize(model: torch.nn.Module, calib: DataLoader, size: int = IMAGE_SIZE) -> torch.nn.Module:
    sample = (torch.randn(1, 3, size, size),)
    graph = torch.export.export_for_training(model, sample).module()
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    move_exported_model_to_eval(prepared)        # calibrate with eval-mode dropout/batch-norm
    seen = 0
    with torch.no_grad():
        for x, _ in calib:
            prepared(x)
            seen += len(x)
            if seen >= CALIBRATION_IMAGES:
                break
    quantized = convert_pt2e(prepared)
    move_exported_model_to_eval(quantized)
    allow_exported_model_train_eval(quantized)   # lets shared helpers call .eval() on it
    return quantized


def to_pte(quantized: torch.nn.Module, size: int = IMAGE_SIZE) -> bytes:
    sample = (torch.randn(1, 3, size, size),)
    exported = torch.export.export(quantized, sample)
    return to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch().buffer


def predict_pte(pte: bytes, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
    module = _load_for_executorch_from_buffer(pte)
    logits, labels, times = [], [], []
    for x, y in loader:
        for i in range(len(x)):                       # the .pte has a fixed batch of 1, like the app
            t0 = time.perf_counter()
            logits.append(module.forward((x[i:i + 1],))[0].numpy())
            times.append(time.perf_counter() - t0)
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels), float(np.median(times) * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    args = ap.parse_args()
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = ckpt_meta(args.ckpt)
    trained_fold = json.loads(torch.load(args.ckpt, map_location="cpu", weights_only=True).get("args", "{}")).get("field_fold")
    model, classes = load_fp32(args.ckpt)
    size = input_size(meta["arch"])
    deployed = Normalized(model).eval()
    rows = read_manifest(taxonomy=meta["taxonomy"])
    sets = {"pv_val": select(rows, {"pv_val"}, classes)}
    field_train: list[dict] = []
    if trained_fold in (0, 1):
        # Parity on the field half this checkpoint never saw; calibration may use the half it trained on.
        field_train, field_held = split_field(rows, classes, trained_fold, meta["fold_scheme"])
        sets["field_heldout"] = field_held
    else:
        print("WARNING: checkpoint trained on all field data; field parity unavailable, use the fold models")

    def loader(part: list[dict], transform) -> DataLoader:
        return DataLoader(ManifestDataset(part, classes, transform), batch_size=64, num_workers=8)

    if args.precision == "int8":
        calib_rows = calibration_rows(select(rows, {"pv_calib"}, classes), field_train)
        graph = quantize(deployed, loader(calib_rows, deploy_transform(size)), size)
    else:
        calib_rows, graph = [], deployed
    pte = to_pte(graph, size)
    pte_path = out / f"tomato_{meta['arch']}_{meta['taxonomy']}_{args.precision}_xnnpack.pte"
    pte_path.write_bytes(pte)

    results = {"classes": classes, "taxonomy": meta["taxonomy"], "precision": args.precision, "pte": str(pte_path),
               "pte_bytes": len(pte), "pte_sha256": hashlib.sha256(pte).hexdigest(), "calibration_images": len(calib_rows)}
    for name, part in sets.items():
        fp32_logits, labels = predict(model, loader(part, eval_transform(size)))
        pte_logits, _, median_ms = predict_pte(pte, loader(part, deploy_transform(size)))
        fp32, on_device = report(fp32_logits, labels, classes), report(pte_logits, labels, classes)
        results[name] = {
            "fp32_macro_f1": fp32["macro_f1"], "pte_macro_f1": on_device["macro_f1"],
            "delta_pte_minus_fp32": on_device["macro_f1"] - fp32["macro_f1"],
            "top1_agreement": float((fp32_logits.argmax(1) == pte_logits.argmax(1)).mean()),
            "max_abs_logit_diff": float(np.abs(fp32_logits - pte_logits).max()),
            "pte_median_ms_x86": median_ms, "n": fp32["n"],
            "per_class_recall_fp32": {c: fp32["per_class"][c]["recall"] for c in classes},
            "per_class_recall_pte": {c: on_device["per_class"][c]["recall"] for c in classes},
        }
    (out / f"{pte_path.stem}_parity.json").write_text(json.dumps(results, indent=1))
    sidecar = {
        "pte_file": pte_path.name, "pte_sha256": results["pte_sha256"], "arch": meta["arch"],
        "taxonomy": meta["taxonomy"], "precision": args.precision, "classes": classes,
        "input": sidecar_input(size),
        "output": "logits; apply temperature + decision rule from the threshold file bound to this sha",
        "checkpoint": args.ckpt, "checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "executorch": getattr(executorch, "__version__", "unknown"), "torch": torch.__version__,
    }
    (out / f"{pte_path.stem}.json").write_text(json.dumps(sidecar, indent=1))
    print(json.dumps({k: results[k] for k in ("precision", "pte_bytes", "pte_sha256")}, indent=1))
    for name in sets:
        r = results[name]
        print(f"{name}: FP32 {r['fp32_macro_f1']:.3f}  .pte {r['pte_macro_f1']:.3f}  (delta {r['delta_pte_minus_fp32']:+.3f}, "
              f"top-1 agreement {r['top1_agreement']:.3f}, n={r['n']}, {r['pte_median_ms_x86']:.1f} ms/img x86)")


if __name__ == "__main__":
    main()
