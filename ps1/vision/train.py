"""W4: fine-tune MobileNetV3-Small on the supported tomato classes (CPU).

Model selection uses pv_val and field_dev only; the sealed test is never read here.
Usage: python -m ps1.vision.train --classes tomato_early_blight,... --epochs 15 --out runs/v0
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    efficientnet_b2,
    mobilenet_v3_small,
)

from ps1.data.build_splits import OTHER_CLASS
from ps1.vision.dataset import (
    MANIFEST,
    TAXONOMIES,
    CompositeDataset,
    ManifestDataset,
    eval_transform,
    group_folds,
    input_size,
    read_manifest,
    select,
    train_transform,
)

ARCHS = ("mnv3s", "effnet_b0", "effnet_b2")


def build_model(num_classes: int, pretrained: bool = True, arch: str = "mnv3s") -> nn.Module:
    if arch == "mnv3s":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "effnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "effnet_b2":
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return model


@torch.no_grad()
@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x))
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def report(logits: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict:
    """Metrics; macro-F1 averages only classes present in `labels` (a class with no rows, e.g. 'other'
    in pv_val, would otherwise count as F1 = 0 and cap the score)."""
    pred = logits.argmax(1)
    idx = list(range(len(classes)))
    present = sorted(set(labels.tolist()))
    return {
        "n": int(len(labels)),
        "macro_f1": float(f1_score(labels, pred, labels=present, average="macro", zero_division=0)) if present else 0.0,
        "accuracy": float((pred == labels).mean()) if len(labels) else 0.0,
        "per_class": classification_report(labels, pred, labels=idx, target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(labels, pred, labels=idx).tolist(),
    }


def field_backgrounds(rows: list[dict]) -> list[str]:
    """PlantDoc photos of *other* crops from other_train: field scenes with no tomato dev/test overlap."""
    return [r["path"] for r in rows if r["split"] == "other_train" and r["source"] == "plantdoc"]


FOLD_SCHEMES = ("roundrobin", "stratified")   # roundrobin = v2/v3; stratified = StratifiedGroupKFold (31 Oct)


def _fold(field: list[dict], fold: int | None, scheme: str = "roundrobin") -> tuple[list[dict], list[dict]]:
    """(rows to train on, rows to select on). fold=None: none in training; -1: all in training."""
    if fold is None:
        return [], field
    if fold == -1:
        return field, []
    labels = [r["source_canonical"] for r in field] if scheme == "stratified" else None
    held_out = group_folds([r["group"] for r in field], labels=labels)[fold]
    return [r for r, h in zip(field, held_out) if not h], [r for r, h in zip(field, held_out) if h]


def split_field(rows: list[dict], classes: list[str], fold: int | None,
                scheme: str = "roundrobin") -> tuple[list[dict], list[dict]]:
    """Field rows of the supported classes, split by group-aware fold."""
    return _fold(select(rows, {"field_dev"}, classes), fold, scheme)


def as_other(rows: list[dict]) -> list[dict]:
    return [{**r, "canonical": OTHER_CLASS} for r in rows]


def split_other(rows: list[dict], fold: int | None,
                scheme: str = "roundrobin") -> tuple[list[dict], list[dict], list[dict]]:
    """Outlier exposure rows: (lab train, field train fold, field select fold).

    Lab = ood_train (PlantVillage). Field = PlantDoc part of ood_dev, folded like field_dev.
    Maharashtra OOD is never used here: it stays an unseen-problem evaluation set.
    """
    field_ood = [r for r in select(rows, {"ood_dev"}) if r["source"] == "plantdoc"]
    field_train, field_select = _fold(field_ood, fold, scheme)
    return as_other(select(rows, {"ood_train"})), as_other(field_train), as_other(field_select)


def training_rows(rows: list[dict], classes: list[str], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """(training rows incl. oversampled field rows, held-out field rows for selection)."""
    field_train, field_select = split_field(rows, classes, args.field_fold, args.fold_scheme)
    excluded = set(getattr(args, "exclude_sources", "").split(",")) - {""}
    lab = [r for r in select(rows, {"pv_train", "ext_train"}, classes) if r["source"] not in excluded]
    train = lab + field_train * args.field_oversample
    if OTHER_CLASS in classes:
        other_lab, other_field_train, other_field_select = split_other(rows, args.field_fold, args.fold_scheme)
        train += other_lab + other_field_train * args.field_oversample
        field_select = field_select + other_field_select
    extra_mult = getattr(args, "extra_oversample", 0)
    if extra_mult > 0:   # 21 Nov: extra field sources (data/manifest_extra.csv); training only, never selection
        train += select(rows, {"x_field_train"}, classes) * extra_mult
        if OTHER_CLASS in classes:
            train += as_other(select(rows, {"x_ood_train"})) * extra_mult
    return train, field_select


def loaders(rows: list[dict], classes: list[str], args: argparse.Namespace,
            train_rows: list[dict], field_select: list[dict]) -> dict[str, DataLoader]:
    def make(ds, sampler=None) -> DataLoader:
        # Only the training loader keeps its workers alive; validation workers exit after each evaluation
        # (persistent eval workers held ~6 idle processes per run and pushed the laptop into OOM).
        is_train = sampler is not None
        return DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                          persistent_workers=is_train and args.workers > 0, drop_last=is_train)
    size = input_size(args.arch)
    train_ds = (CompositeDataset(train_rows, classes, train_transform(size), field_backgrounds(rows), args.composite_p)
                if args.composite_p > 0 else ManifestDataset(train_rows, classes, train_transform(size)))
    n = args.samples_per_epoch or len(train_ds)
    out = {"train": make(train_ds, RandomSampler(train_ds, replacement=False, num_samples=min(n, len(train_ds)))),
           "pv_val": make(ManifestDataset(select(rows, {"pv_val"}, classes), classes, eval_transform(size)))}
    if field_select:
        out["field_select"] = make(ManifestDataset(field_select, classes, eval_transform(size)))
    return out


def class_weights(train_rows: list[dict], classes: list[str]) -> torch.Tensor:
    """Inverse-frequency weights over the actual training list (the 'other' class is large)."""
    counts = Counter(r["canonical"] for r in train_rows)
    w = torch.tensor([1.0 / max(counts[c], 1) for c in classes])
    return w / w.sum() * len(classes)


def checkpoint(model: nn.Module, classes: list[str], epoch: int, args: argparse.Namespace) -> dict:
    """Weights plus everything needed to say what produced them (weights_only-loadable types only)."""
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    extra = getattr(args, "extra_manifest", "")
    extra_sha = hashlib.sha256(Path(extra).read_bytes()).hexdigest() if extra else ""
    return {"extra_manifest_sha256": extra_sha, "state_dict": model.state_dict(), "classes": classes, "epoch": epoch, "arch": args.arch,
            "taxonomy": args.taxonomy, "fold_scheme": args.fold_scheme,
            "versions": json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__}),
            "args": json.dumps(vars(args), sort_keys=True), "manifest_sha256": manifest_sha}


WEIGHT_DECAY = 1e-4


def make_ema(model: nn.Module, decay: float) -> AveragedModel | None:
    """Exponential moving average of the weights AND BatchNorm buffers (Option A, 21 Nov); None when off.

    The first update_parameters() copies the model; later ones blend: ema = decay * ema + (1 - decay) * model.
    """
    if decay <= 0.0:
        return None
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay), use_buffers=True)


def param_groups(model: nn.Module, lrs: list[float]) -> list[dict]:
    """Backbone / head at their own LRs; biases and norm parameters (1-D tensors) get no weight decay."""
    head_ids = {id(p) for p in model.classifier.parameters()}
    groups = []
    for is_head, lr in ((False, lrs[0]), (True, lrs[1])):
        params = [(n, p) for n, p in model.named_parameters() if (id(p) in head_ids) == is_head]
        groups.append({"params": [p for _, p in params if p.ndim > 1], "lr": lr})
        groups.append({"params": [p for _, p in params if p.ndim <= 1], "lr": lr, "weight_decay": 0.0})
    return [g for g in groups if g["params"]]


def validate_rows(rows: list[dict], classes: list[str], field_select: list[dict]) -> None:
    """Fail at start-up, not mid-epoch: every image exists and every class has training rows."""
    missing = [r["path"] for r in rows if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images missing, e.g. {missing[:3]}")
    counts = Counter(r["canonical"] for r in rows)
    empty = [c for c in classes if counts[c] == 0]
    if empty:
        raise ValueError(f"classes with no rows: {empty}")
    if field_select:
        held = Counter(r["canonical"] for r in field_select)
        thin = [c for c in classes if held[c] == 0]
        if thin:
            print(f"WARNING: no held-out field rows for {thin}; their field F1 is not measured")


def atomic_save(obj: dict, path: Path) -> None:
    """Write to a temp file then rename, so a killed run never leaves a half-written checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True, help="comma-separated canonical classes, in output order")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--composite-p", type=float, default=0.0, help="prob. of pasting the leaf on a field background")
    ap.add_argument("--backbone-lr-mult", type=float, default=1.0, help="backbone LR = lr * this")
    ap.add_argument("--arch", choices=ARCHS, default="mnv3s")
    ap.add_argument("--field-fold", type=int, default=None,
                    help="0/1: train on the other field_dev fold, select on this one; -1: train on all field_dev")
    ap.add_argument("--field-oversample", type=int, default=1, help="repeat field training rows this many times")
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = whole training set each epoch")
    ap.add_argument("--taxonomy", choices=sorted(TAXONOMIES), default="t1", help="label map applied on read")
    ap.add_argument("--fold-scheme", choices=FOLD_SCHEMES, default="stratified")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema-decay", type=float, default=0.0, help="0 = off; else also save ema.pt (weights + BN stats)")
    ap.add_argument("--ema-start-epoch", type=int, default=4, help="first epoch whose steps update the EMA")
    ap.add_argument("--exclude-sources", default="",
                    help="comma list of sources left out of training (ablations), e.g. maharashtra")
    ap.add_argument("--extra-manifest", default="", help="data/manifest_extra.csv: extra field sources (21 Nov)")
    ap.add_argument("--extra-oversample", type=int, default=0, help="repeat extra training rows; 0 = off")
    args = ap.parse_args()
    if bool(args.extra_manifest) != (args.extra_oversample > 0):
        ap.error("--extra-manifest and --extra-oversample > 0 go together")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    classes = args.classes.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(taxonomy=args.taxonomy)
    if args.extra_manifest:
        rows += read_manifest(args.extra_manifest, taxonomy=args.taxonomy)
    train_rows, field_select = training_rows(rows, classes, args)
    validate_rows(train_rows + field_select, classes, field_select)
    dl = loaders(rows, classes, args, train_rows, field_select)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    model = build_model(len(classes), arch=args.arch)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_rows, classes), label_smoothing=0.05)
    lrs = [args.lr * args.backbone_lr_mult, args.lr]
    opt = torch.optim.AdamW(param_groups(model, lrs), weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=args.epochs * len(dl["train"]))

    ema = make_ema(model, args.ema_decay)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running = time.time(), 0.0
        for x, y in dl["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            if ema is not None and epoch >= args.ema_start_epoch:
                ema.update_parameters(model)
            running += loss.item() * len(y)
        val = report(*predict(model, dl["pv_val"]), classes)
        dev = report(*predict(model, dl["field_select"]), classes) if "field_select" in dl else None
        entry = dict(epoch=epoch, train_loss=running / len(dl["train"].sampler), pv_val_f1=val["macro_f1"],
                     field_f1=dev["macro_f1"] if dev else None, seconds=round(time.time() - start, 1))
        history.append(entry)
        print(json.dumps(entry), flush=True)
        # Select on held-out field data (lab val is near-saturated on PlantVillage and says little).
        # With all field data in training (final run) there is nothing to select on: keep the last epoch.
        score = dev["macro_f1"] if dev else epoch
        if score > best:
            best = score
            atomic_save(checkpoint(model, classes, epoch, args), out / "best.pt")
            (out / "best_metrics.json").write_text(json.dumps({"epoch": epoch, "pv_val": val, "field_select": dev}, indent=1))
    # Fixed-epoch checkpoint: unbiased for CV reporting (best.pt is chosen on the fold it is scored on).
    atomic_save(checkpoint(model, classes, args.epochs, args), out / "last.pt")
    if ema is not None:
        ema_model = ema.module.eval()
        val = report(*predict(ema_model, dl["pv_val"]), classes)
        dev = report(*predict(ema_model, dl["field_select"]), classes) if "field_select" in dl else None
        entry = dict(epoch=args.epochs, ema=True, pv_val_f1=val["macro_f1"], field_f1=dev["macro_f1"] if dev else None)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        atomic_save(checkpoint(ema_model, classes, args.epochs, args), out / "ema.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best selection score {best:.3f}; checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
