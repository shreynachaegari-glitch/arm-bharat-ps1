"""Extra field sources (21 Nov, Sreyesh: "find more datasets ... and try training them") -> data/manifest_extra.csv.

The main manifest (data/manifest.csv) is pinned by the sealed run (manifest sha in data/SEALED_RUN_STARTED.json),
so it is never rebuilt for this. New sources get their own manifest:

  x_field_train   supported classes from the training sources: extra field images for training
  x_ood_train     unsupported problems from the training sources: trains the 'other problem' class
  x_test_known    supported classes from the held-out source: a fresh cross-source external test
  x_test_ood      unsupported problems from the held-out source
  x_dev_known     22 Nov (Sreyesh): half of the held-out source, by duplicate group, used to choose between
  x_dev_ood       fine-tuning variants (v9); the x_test_* half stays untouched until the final check
  excluded        duplicates of ANY main-manifest image (sealed ones included), label conflicts,
                  fruit images, uncategorised folders

The held-out source never enters training or model selection; it is scored after training only.
Usage: python -m ps1.data.build_extra
"""
import csv
import hashlib
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ps1.data.build_splits import DHASH_CONFIRM_BITS, NEAR_DUP_BITS, group_duplicates, hash_image
from ps1.data.taxonomy import IMAGE_SUFFIXES

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data/raw"
MAIN_MANIFEST = REPO_ROOT / "data/manifest.csv"
OUT = REPO_ROOT / "data/manifest_extra.csv"

# source -> (folder under data/raw, role, licence/provenance)
SOURCES = {
    "bd_diu_2025a": ("bd_diu_2025a", "train", "CC-BY-4.0, Mendeley jttrv2w27r v2 (Daffodil Int. Univ., Bangladesh field)"),
    "bd_diu_2025b": ("bd_diu_2025b", "train", "CC-BY-4.0, Mendeley 93h9p62kg4 v1 (Daffodil Int. Univ., Bangladesh field)"),
    "bd_neu_2026": ("bd_neu_2026/extracted", "test", "CC-BY-4.0, Mendeley 9jxvtgh325 v1 (NEU Bangladesh field), originals only"),
}

# normalised folder label -> canonical (t1 names). Anything not listed aborts the build, so each new label is reviewed.
LABELS = {
    "bacteiral spot": "tomato_bacterial_spot", "bacterial spot": "tomato_bacterial_spot",
    "early blight": "tomato_early_blight", "early blight leaf": "tomato_early_blight",
    "late blight": "tomato_late_blight", "late blight leaf": "tomato_late_blight",
    "healthy": "tomato_healthy", "healthy leaf": "tomato_healthy", "healthy leaves": "tomato_healthy",
    "leaf mold": "tomato_leaf_mold",
    "yellow leaf curl virus": "tomato_yellow_leaf_curl_virus", "leaf curl virus": "tomato_yellow_leaf_curl_virus",
    "tylcv": "tomato_yellow_leaf_curl_virus", "leaf curl leaf": "tomato_yellow_leaf_curl_virus",
    "leaf miner": "UNSUPPORTED", "leaf miner flies": "UNSUPPORTED", "insect damage": "UNSUPPORTED",
    "spider mites": "UNSUPPORTED", "magnesium deficiency": "UNSUPPORTED", "cercospora leaf mold": "UNSUPPORTED",
    "bushy stunt leaf": "UNSUPPORTED",
    "healthy fruit": "EXCLUDED", "healthy fruits": "EXCLUDED", "target spot fruit": "EXCLUDED",
    "target spot fruits": "EXCLUDED", "uncategorized": "EXCLUDED", "other": "EXCLUDED",
}
SUPPORTED_T1 = {"tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight", "tomato_septoria_leaf_spot",
                "tomato_healthy"}
FIELDS = ["path", "source", "source_split", "source_label", "canonical", "split", "group", "sha256", "phash",
          "dhash", "exclude_reason"]


def normalise(label: str) -> str:
    s = re.sub(r"[_\-]+", " ", label.lower())
    s = re.sub(r"^(tomato|class \d+)\s*", "", s.strip())      # "Tomato Early Blight", "Class 16 (Uncategorized)"
    s = re.sub(r"[()]", " ", s)
    return " ".join(s.split())


def canonical_for(label: str) -> str:
    n = normalise(label)
    return LABELS.get(n, LABELS.get(n.removesuffix("s"), "UNMAPPED"))


def collect(raw: Path = RAW) -> list[dict]:
    rows = []
    for source, (sub, role, _lic) in SOURCES.items():
        root = raw / sub
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or "augment" in str(path.relative_to(root)).lower():
                continue
            label = path.parent.name
            rows.append(dict(path=str(path.relative_to(REPO_ROOT)), source=source, source_split=role,
                             source_label=label, canonical=canonical_for(label)))
    return rows


def split_for(row: dict) -> tuple[str, str]:
    c, role = row["canonical"], row["source_split"]
    if c == "EXCLUDED":
        return "excluded", "not a supported leaf image (fruit or uncategorised)"
    supported = c in SUPPORTED_T1
    if role == "test":
        return ("x_test_known" if supported else "x_test_ood"), ""
    return ("x_field_train" if supported else "x_ood_train"), ""


def near_main(new_ph: np.ndarray, new_dh: np.ndarray, main_ph: np.ndarray, main_dh: np.ndarray,
              chunk: int = 1024) -> np.ndarray:
    """Index of the first near-duplicate main row for each new row, or -1."""
    hit = np.full(len(new_ph), -1, dtype=np.int64)
    if len(main_ph) == 0:
        return hit
    for s in range(0, len(new_ph), chunk):
        near = (np.bitwise_count(new_ph[s:s + chunk, None] ^ main_ph[None, :]) <= NEAR_DUP_BITS) & \
               (np.bitwise_count(new_dh[s:s + chunk, None] ^ main_dh[None, :]) <= DHASH_CONFIRM_BITS)
        any_hit = near.any(axis=1)
        hit[s:s + chunk][any_hit] = near[any_hit].argmax(axis=1)
    return hit


def assign(rows: list[dict], main: list[dict]) -> None:
    """Fill split / exclude_reason / group. `main` rows need sha256, phash, dhash, split."""
    main_sha = {r["sha256"]: r["split"] for r in main}
    usable = [r for r in main if r["phash"] and r["dhash"]]
    main_ph = np.array([int(r["phash"], 16) for r in usable], dtype=np.uint64)
    main_dh = np.array([int(r["dhash"], 16) for r in usable], dtype=np.uint64)
    readable = [r for r in rows if r["phash"]]
    hits = near_main(np.array([int(r["phash"], 16) for r in readable], dtype=np.uint64),
                     np.array([int(r["dhash"], 16) for r in readable], dtype=np.uint64), main_ph, main_dh) \
        if readable else []
    near = {id(r): usable[h]["split"] for r, h in zip(readable, hits, strict=True) if h >= 0}
    groups = group_duplicates([r["sha256"] for r in rows], [r["phash"] for r in rows], [r["dhash"] for r in rows])
    by_group: dict[int, set] = {}
    for r, g in zip(rows, groups, strict=True):
        r["group"] = f"x{g}"
        by_group.setdefault(g, set()).add((r["canonical"], r["source_split"]))
    smallest_sha: dict[int, str] = {}
    for r, g in zip(rows, groups, strict=True):
        smallest_sha[g] = min(smallest_sha.get(g, r["sha256"]), r["sha256"])
    dev_half = {g: hashlib.sha256(s.encode()).digest()[0] % 2 == 0 for g, s in smallest_sha.items()}
    for r, g in zip(rows, groups, strict=True):
        r["split"], r["exclude_reason"] = split_for(r)
        if r["split"] == "excluded":
            continue
        if not r["phash"]:
            r["split"], r["exclude_reason"] = "excluded", "unreadable"
        elif r["sha256"] in main_sha:
            r["split"], r["exclude_reason"] = "excluded", f"exact copy of main-manifest image ({main_sha[r['sha256']]})"
        elif id(r) in near:
            r["split"], r["exclude_reason"] = "excluded", f"near-duplicate of main-manifest image ({near[id(r)]})"
        elif len({c for c, _ in by_group[g]}) > 1:
            r["split"], r["exclude_reason"] = "excluded", "near-duplicates with conflicting labels"
        elif r["source_split"] == "train" and any(role == "test" for _, role in by_group[g]):
            r["split"], r["exclude_reason"] = "excluded", "near-duplicate of held-out test image"
        elif r["split"].startswith("x_test_") and dev_half[g]:       # whole duplicate groups move together
            r["split"] = r["split"].replace("x_test_", "x_dev_")


def main() -> int:
    rows = collect()
    unmapped = Counter((r["source"], r["source_label"]) for r in rows if r["canonical"] == "UNMAPPED")
    if unmapped:
        print("unmapped folder labels (add them to LABELS after looking at the images):", dict(unmapped))
        return 1
    with Pool() as p:
        hashes = p.map(hash_image, [str(REPO_ROOT / r["path"]) for r in rows], chunksize=32)
    for r, (s, ph, dh) in zip(rows, hashes, strict=True):
        r["sha256"], r["phash"], r["dhash"] = s, ph, dh
    with open(MAIN_MANIFEST, newline="") as f:
        main_rows = list(csv.DictReader(f))
    assign(rows, main_rows)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images -> {OUT.relative_to(REPO_ROOT)}")
    print("excluded:", dict(Counter(r["exclude_reason"] for r in rows if r["split"] == "excluded")))
    for (src, split, c), n in sorted(Counter((r["source"], r["split"], r["canonical"]) for r in rows).items()):
        print(f"  {src:14} {split:14} {c:32} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
