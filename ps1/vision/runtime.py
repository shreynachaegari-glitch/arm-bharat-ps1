"""On-device vision inference: the exported .pte plus its sidecar contract (the code the Pi runs).

The sidecar (written by ps1.vision.export) fixes class order and preprocessing; this module refuses a
.pte whose hash differs from the sidecar's, so a model can never run with another model's labels.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from torchvision import transforms as T

from ps1.vision.dataset import INPUT_SIZE, open_rgb, resize_for

SIZES = frozenset(INPUT_SIZE.values())


def input_spec(sidecar_input: dict) -> tuple[int, int]:
    """(crop size, shorter-side resize) from a sidecar's "input"; sidecars before 23 Nov have no "resize" (256)."""
    shape = sidecar_input.get("shape")
    if not (isinstance(shape, list) and len(shape) == 4 and shape[:2] == [1, 3] and shape[2] == shape[3]
            and shape[2] in SIZES):
        raise ValueError(f"unexpected input shape {shape}")
    size = shape[2]
    return size, int(sidecar_input.get("resize", resize_for(size)))


@dataclass(frozen=True)
class VisionModel:
    classes: tuple[str, ...]
    pte_sha256: str
    taxonomy: str
    precision: str
    module: object
    size: int = 224
    resize: int = 256

    def preprocess(self, image_path: str | Path) -> torch.Tensor:
        """EXIF-aware decode -> resize -> centre-crop (sizes from the sidecar) -> [0, 1] float, NCHW."""
        image = open_rgb(image_path)
        tensor = T.Compose([T.Resize(self.resize), T.CenterCrop(self.size), T.ToTensor()])(image)
        return tensor.unsqueeze(0)

    def logits(self, batch: torch.Tensor) -> np.ndarray:
        return self.module.forward((batch,))[0].numpy()[0]


def load_vision(pte_path: str | Path, sidecar_path: str | Path | None = None) -> VisionModel:
    pte_path = Path(pte_path)
    sidecar_path = Path(sidecar_path) if sidecar_path else pte_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    blob = pte_path.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != sidecar["pte_sha256"]:
        raise ValueError(f"{pte_path.name} sha256 {sha[:12]} does not match its sidecar ({sidecar['pte_sha256'][:12]})")
    size, resize = input_spec(sidecar["input"])
    # Load from the file, not from `blob`: the from-buffer loader does not copy the bytes, so the program
    # would point at freed memory once `blob` goes out of scope (it segfaulted in forward()).
    return VisionModel(tuple(sidecar["classes"]), sha, sidecar.get("taxonomy", "t1"), sidecar.get("precision", "int8"),
                       _load_for_executorch(str(pte_path)), size, resize)
