"""Checkpoint loading and metadata, shared by evaluation and export (no ExecuTorch import here)."""
import json
from pathlib import Path

import torch

from ps1.vision.train import build_model


def _load(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def ckpt_meta(path: str | Path) -> dict:
    """What produced a checkpoint. v0-v3 predate taxonomy/fold_scheme (t1, roundrobin)."""
    ckpt = _load(path)
    args = json.loads(ckpt.get("args", "{}"))
    return {"classes": ckpt["classes"], "arch": ckpt.get("arch", "mnv3s"), "epoch": ckpt.get("epoch"),
            "taxonomy": ckpt.get("taxonomy", "t1"), "fold_scheme": ckpt.get("fold_scheme", "roundrobin"),
            "field_fold": args.get("field_fold"), "seed": args.get("seed", 0)}


def load_fp32(path: str | Path) -> tuple[torch.nn.Module, list[str]]:
    ckpt = _load(path)
    model = build_model(len(ckpt["classes"]), pretrained=False, arch=ckpt.get("arch", "mnv3s"))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["classes"]


def check_fold_checkpoints(paths: list[str]) -> dict:
    """Fold checkpoints must agree on what they predict and be passed in fold order (fold i at index i).

    Swapped or duplicated arguments would score each model on its own training half without any error.
    Returns the shared metadata.
    """
    metas = [ckpt_meta(p) for p in paths]
    for key in ("classes", "taxonomy", "fold_scheme", "arch"):
        if len({json.dumps(m[key]) for m in metas}) != 1:
            raise ValueError(f"fold checkpoints disagree on {key}: {[m[key] for m in metas]}")
    for i, m in enumerate(metas):
        if m["field_fold"] is not None and m["field_fold"] != i:
            raise ValueError(f"checkpoint {paths[i]} was trained as fold {m['field_fold']}, passed as fold {i}")
    return metas[0]
