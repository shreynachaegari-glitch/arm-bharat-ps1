"""Manifest-backed image dataset with a hard guard on the sealed final test (v4 P0-1)."""
import csv
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage import binary_fill_holes
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms as T

from ps1.data.build_splits import SEALED_SPLITS as BUILD_SEALED_SPLITS

SEALED_SPLITS = frozenset(BUILD_SEALED_SPLITS)     # single source of truth: ps1/data/build_splits.py
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data/manifest.csv"
# Training-time label maps. The manifest keeps the original labels (so splits, incl. the sealed test,
# never move); a taxonomy relabels rows on read and is recorded in every checkpoint.
#   t1: 4 diseases + healthy (30 Oct).  t2: bacterial spot + Septoria merged into a visual leaf-spot
#   group (Sreyesh 31 Oct): the model confused them (21/54, 13/75), so it must not claim to tell them apart.
TAXONOMIES: dict[str, dict[str, str]] = {
    "t1": {},
    "t2": {"tomato_bacterial_spot": "tomato_leaf_spot", "tomato_septoria_leaf_spot": "tomato_leaf_spot"},
}
IMAGE_SIZE = 224             # default and every model up to v8
# Input size per architecture (23 Nov): EfficientNet-B2 runs at 260 px (~2.6x B0's compute, within the 3x Pi budget).
INPUT_SIZE = {"mnv3s": 224, "effnet_b0": 224, "effnet_b2": 260}


def input_size(arch: str) -> int:
    if arch not in INPUT_SIZE:
        raise ValueError(f"no input size for arch {arch!r}")
    return INPUT_SIZE[arch]


def resize_for(size: int) -> int:
    """Shorter-side resize before the centre crop, same crop ratio as 256 -> 224."""
    return round(size * 256 / 224)

MASK_THRESHOLD = 18          # segmented PlantVillage leaves sit on near-black pixels
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # ImageNet stats (pretrained backbone)


def read_manifest(path: str | Path = MANIFEST, taxonomy: str = "t1") -> list[dict]:
    """Manifest rows with absolute image paths and `canonical` mapped by `taxonomy`.

    `source_canonical` keeps the original label (used for fold stratification, so t1 and t2 runs
    share identical folds).
    """
    mapping = TAXONOMIES[taxonomy]
    with open(path, newline="") as f:
        return [{**r, "path": str(REPO_ROOT / r["path"]), "source_canonical": r["canonical"],
                 "canonical": mapping.get(r["canonical"], r["canonical"])} for r in csv.DictReader(f)]


def open_rgb(path: str | Path) -> Image.Image:
    """Decode with EXIF orientation applied: phone photos are often stored rotated."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def select(rows: list[dict], splits: set[str], classes: list[str] | None = None) -> list[dict]:
    """Rows for the requested splits. Refuses sealed splits unless PS1_FINAL_EVAL=1 is set."""
    sealed = splits & SEALED_SPLITS
    if sealed and os.environ.get("PS1_FINAL_EVAL") != "1":
        raise PermissionError(f"{sorted(sealed)} sealed; set PS1_FINAL_EVAL=1 only for the single final run")
    return [r for r in rows if r["split"] in splits and (classes is None or r["canonical"] in classes)]


def group_folds(groups: list[str], k: int = 2, seed: int = 0, labels: list[str] | None = None) -> list[list[bool]]:
    """k boolean masks over `groups`; a dedup group always lands in exactly one fold.

    With `labels`, folds are class-stratified (StratifiedGroupKFold). Without, the original
    round-robin scheme used for v2/v3 (kept so those checkpoints are evaluated on their own folds).
    """
    if labels is not None:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        masks = [[False] * len(groups) for _ in range(k)]
        for i, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(groups)), labels, groups)):
            for j in test_idx:
                masks[i][j] = True
        return masks
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of = {g: i % k for i, g in enumerate(unique)}
    return [[fold_of[g] == i for g in groups] for i in range(k)]


EVAL_BATCH, EVAL_WORKERS = 64, 8


def eval_loader(rows: list[dict], classes: list[str], transform: Callable | None = None,
                size: int = IMAGE_SIZE) -> "torch.utils.data.DataLoader":
    """The one evaluation DataLoader (no shuffling, no persistent workers)."""
    return torch.utils.data.DataLoader(ManifestDataset(rows, classes, transform or eval_transform(size)),
                                       batch_size=EVAL_BATCH, num_workers=EVAL_WORKERS)


def train_transform(size: int = IMAGE_SIZE) -> Callable:
    """Field-style augmentation: framing, rotation, lighting and blur that PlantVillage lacks (v4 2.6)."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.35, hue=0.04),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


def eval_transform(size: int = IMAGE_SIZE) -> Callable:
    return T.Compose([T.Resize(resize_for(size)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def segmented_path(color_path: str) -> Path:
    """PlantVillage raw/color/<cls>/<name>.JPG -> raw/segmented/<cls>/<name>_final_masked.jpg."""
    p = Path(color_path)
    return Path(str(p.parent).replace("/raw/color/", "/raw/segmented/")) / f"{p.stem}_final_masked.jpg"


def composite_on_background(leaf_rgb: Image.Image, background: Image.Image, rng: random.Random) -> Image.Image:
    """Paste a segmented leaf (black backdrop) onto a random crop of a field photo (v4 2.6)."""
    leaf = np.asarray(leaf_rgb)
    # Dark necrotic lesions fall under the threshold too; fill enclosed holes so disease pixels are kept
    # instead of being replaced by background texture.
    mask = binary_fill_holes(leaf.max(axis=2) > MASK_THRESHOLD).astype(np.uint8) * 255
    alpha = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(1.5))     # soften the cut edge
    w, h = leaf_rgb.size
    bg = background.convert("RGB")
    scale = rng.uniform(1.0, 2.0)                                           # zoom so the crop isn't the whole scene
    bg = bg.resize((max(w, int(bg.width * scale * w / max(bg.width, 1))),
                    max(h, int(bg.height * scale * h / max(bg.height, 1)))))
    x0, y0 = rng.randint(0, bg.width - w), rng.randint(0, bg.height - h)
    canvas = bg.crop((x0, y0, x0 + w, y0 + h))
    canvas.paste(leaf_rgb, (0, 0), alpha)
    return canvas


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], classes: list[str], transform: Callable) -> None:
        self.rows = rows
        self.index = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        return self.transform(open_rgb(r["path"])), self.index[r["canonical"]]


class CompositeDataset(ManifestDataset):
    """Training set that, with probability p, replaces the PlantVillage backdrop by a field background."""

    def __init__(self, rows: list[dict], classes: list[str], transform: Callable,
                 backgrounds: list[str], p: float) -> None:
        super().__init__(rows, classes, transform)
        self.backgrounds, self.p = backgrounds, p
        # Rows that can actually be composited (PlantVillage with a segmented twin); others always fall back.
        eligible = sum(r["source"] == "plantvillage" and segmented_path(r["path"]).exists() for r in rows)
        self.effective_p = p * eligible / max(len(rows), 1)
        print(f"composite: {eligible}/{len(rows)} rows eligible -> effective composite rate {self.effective_p:.2f} "
              f"(requested {p:.2f})")

    def __getitem__(self, i: int):
        r = self.rows[i]
        # Drawn from torch's RNG, which torch.manual_seed + DataLoader seed per worker and epoch:
        # reproducible across runs, and workers don't share one stream.
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,))))
        seg = segmented_path(r["path"])
        if self.backgrounds and r["source"] == "plantvillage" and seg.exists() and rng.random() < self.p:
            img = composite_on_background(open_rgb(seg), open_rgb(rng.choice(self.backgrounds)), rng)
        else:
            img = open_rgb(r["path"])
        return self.transform(img), self.index[r["canonical"]]
