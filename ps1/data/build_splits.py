"""Global dedup + group-aware splits -> data/manifest.csv (v4 P0-1, 2.3; addenda #1).

Splits (SEALED = read once, after the model, thresholds and preprocessing are frozen)
  pv_train / pv_val / pv_calib   PlantVillage, supported tomato classes (lab-style development data)
  field_dev                      PlantDoc supported tomato, 50% of pooled train+test: model selection
  final_external_test  SEALED    PlantDoc supported tomato, the other 50%
  ood_train                      unsupported tomato, PlantVillage (lab) part of the former ood_dev:
                                 trains the "other tomato problem" class (outlier exposure)
  ood_dev / ood_test   SEALED    unsupported tomato diseases/pests, field sources (+ PV in ood_test)
  other_train / other_dev / other_test  SEALED(test)   other crops, input-gate "other_plant"

Dedup is global across both datasets: exact sha256, then perceptual-hash near-duplicates
(64-bit pHash, LSH over 4 x 16-bit bands, Hamming <= NEAR_DUP_BITS, confirmed by dHash). Near-duplicate images
share a group and a group never straddles splits: the sealed split keeps it, others drop it.

Usage: python -m ps1.data.build_splits [data_root]
"""
import csv
import hashlib
import random
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from ps1.data.taxonomy import IMAGE_SUFFIXES, MAPPING

NEAR_DUP_BITS = 6          # pHash Hamming distance
DHASH_CONFIRM_BITS = 10    # second, independent hash must also agree (stops transitive chaining:
                           # pHash alone merged 2,636 low-texture PlantVillage images into one group)
BANDS = 4
SEED = 20261030
# Sreyesh 2026-10-30: ship 4 diseases + healthy; other mapped tomato classes become unsupported-disease OOD.
SUPPORTED = ("tomato_bacterial_spot", "tomato_early_blight", "tomato_late_blight",
             "tomato_septoria_leaf_spot", "tomato_healthy")
OTHER_CLASS = "tomato_other_problem"   # outlier-exposure class: "a tomato problem this assistant doesn't cover"
PV_FRACTIONS = (("pv_train", 0.70), ("pv_val", 0.15), ("pv_calib", 0.15))
# Sreyesh 2026-10-30: PlantDoc's own test has 6-11 imgs/class, so pool PD train+test and re-split 50/50.
FIELD_FRACTIONS = (("field_dev", 0.5), ("final_external_test", 0.5))
OOD_FRACTIONS = (("ood_dev", 0.5), ("ood_test", 0.5))
OTHER_FRACTIONS = (("other_train", 0.70), ("other_dev", 0.15), ("other_test", 0.15))
SEALED_SPLITS = ("final_external_test", "ood_test", "other_test")
# When near-duplicates land in different splits, the earliest split here keeps the group.
PRIORITY = (*SEALED_SPLITS, "field_dev", "ood_dev", "other_dev", "pv_val", "pv_calib", "pv_train", "ext_train",
            "ood_train", "other_train")
SEALED = "final_external_test"
FOLDED_SPLITS = ("field_dev", "ood_dev")   # split into group-aware CV folds by the trainer
BASE_SOURCES = ("plantvillage", "plantdoc")
EXT_OOD_FRACTIONS = (("ood_dev", 0.5), ("ood_test", 0.5))


def collect(root: Path) -> list[dict]:
    """One row per image with its source, label and canonical class."""
    rows = []
    folders = [("plantvillage", "all", d) for d in sorted((root / "plantvillage/raw/color").glob("*/"))]
    folders += [("plantdoc", s, d) for s in ("train", "test") for d in sorted((root / "plantdoc" / s).glob("*/"))]
    # Sources added later go last so existing row indices (and hence group ids) never change.
    folders += [("maharashtra", "all", d) for d in sorted((root / "maharashtra/Real world data").glob("*/"))]
    for source, split, folder in folders:
        canonical = MAPPING.get((source, folder.name), ("OTHER_CROP",))[0]
        if canonical == "OTHER_CROP" and folder.name.lower().startswith("tomato"):
            canonical = "UNMAPPED"
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(dict(path=str(path), source=source, source_split=split,
                                 source_label=folder.name, canonical=canonical))
    return rows


def hash_image(path: str) -> tuple[str, str, str]:
    data = Path(path).read_bytes()
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            phash, dhash = str(imagehash.phash(rgb)), str(imagehash.dhash(rgb))
    except (OSError, ValueError, Image.DecompressionBombError):
        phash = dhash = ""               # unreadable or oversized: excluded later, never kills the pool
    return hashlib.sha256(data).hexdigest(), phash, dhash


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _bits(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def group_duplicates(sha: list[str], phash: list[str], dhash: list[str] | None = None) -> list[int]:
    uf = UnionFind(len(sha))
    by_sha: dict[str, int] = {}
    for i, s in enumerate(sha):
        if s in by_sha:
            uf.union(i, by_sha[s])
        else:
            by_sha[s] = i
    ints = [int(p, 16) if p else None for p in phash]
    dints = [int(d, 16) if d else None for d in dhash] if dhash else None
    band_width = 64 // BANDS
    for band in range(BANDS):
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, v in enumerate(ints):
            if v is not None:
                buckets[(v >> (band * band_width)) & ((1 << band_width) - 1)].append(i)
        for members in buckets.values():
            for a_idx, a in enumerate(members):
                for b in members[a_idx + 1:]:
                    if _bits(ints[a], ints[b]) > NEAR_DUP_BITS:
                        continue
                    if dints is None or _bits(dints[a], dints[b]) <= DHASH_CONFIRM_BITS:
                        uf.union(a, b)
    return [uf.find(i) for i in range(len(sha))]


def split_groups(groups: list[int], fractions: tuple[tuple[str, float], ...], rng: random.Random) -> dict[int, str]:
    unique = sorted(set(groups))
    rng.shuffle(unique)
    out, start = {}, 0
    for k, (name, frac) in enumerate(fractions):
        end = len(unique) if k == len(fractions) - 1 else start + round(frac * len(unique))
        out.update({g: name for g in unique[start:end]})
        start = end
    return out


def is_supported(r: dict) -> bool:
    return r["canonical"] in SUPPORTED


def is_ood(r: dict) -> bool:
    if r["canonical"] == "MIXED":
        return False
    return r["canonical"] == "UNSUPPORTED" or (r["canonical"].startswith("tomato_") and not is_supported(r))


def assign(rows: list[dict]) -> None:
    """Fill r["split"] / r["exclude_reason"]; rows need canonical, source, group, phash."""
    rng = random.Random(SEED)
    for r in rows:
        r["split"], r["exclude_reason"] = "excluded", "" if r["phash"] else "unreadable"
    def base(r: dict) -> bool:
        return r["source"] in BASE_SOURCES

    pools = [  # (predicate, fractions, stratify by class); order and RNG stream are frozen for base sources
        (lambda r: r["source"] == "plantdoc" and is_supported(r), FIELD_FRACTIONS, True),
        (lambda r: r["source"] == "plantvillage" and is_supported(r), PV_FRACTIONS, True),
        (lambda r: base(r) and is_ood(r), OOD_FRACTIONS, False),
        (lambda r: base(r) and r["canonical"] == "OTHER_CROP", OTHER_FRACTIONS, False),
    ]
    _assign_pools(rows, pools, rng)
    # Added sources (Sreyesh 30 Oct: "add them"): supported images train only; OOD split dev/test.
    # Separate RNG so adding sources never moves a base-source assignment.
    ext_pools = [
        (lambda r: not base(r) and is_supported(r), (("ext_train", 1.0),), True),
        (lambda r: not base(r) and is_ood(r), EXT_OOD_FRACTIONS, False),
    ]
    _assign_pools(rows, ext_pools, random.Random(SEED + 1))
    _designate_ood_train(rows)          # before the straddle check, so PV/field twins in ood_dev get resolved
    _resolve_straddling_groups(rows)


def _designate_ood_train(rows: list[dict]) -> None:
    """Outlier exposure (Sreyesh 31 Oct): lab OOD dev images train the 'other tomato problem' class.

    A pure relabel after all splits are final, so no other assignment moves. Field OOD
    (PlantDoc, Maharashtra) stays in ood_dev; sealed ood_test is untouched.
    """
    for r in rows:
        if r["split"] == "ood_dev" and r["source"] == "plantvillage":
            r["split"] = "ood_train"


def _assign_pools(rows: list[dict], pools: list, rng: random.Random) -> None:
    for pred, fractions, stratify in pools:
        members = [r for r in rows if r["phash"] and pred(r)]
        strata = sorted({r["canonical"] for r in members}) if stratify else [None]
        for stratum in strata:
            part = [r for r in members if stratum is None or r["canonical"] == stratum]
            mapping = split_groups([r["group"] for r in part], fractions, rng)
            for r in part:
                r["split"] = mapping[r["group"]]


def _resolve_straddling_groups(rows: list[dict]) -> None:
    """Near-duplicates must not straddle splits: the highest-priority split (sealed first) keeps the group."""
    rank = {s: i for i, s in enumerate(PRIORITY)}
    best: dict[int, str] = {}
    for r in rows:
        if r["split"] in rank and (r["group"] not in best or rank[r["split"]] < rank[best[r["group"]]]):
            best[r["group"]] = r["split"]
    for r in rows:
        if r["split"] in rank and r["split"] != best[r["group"]]:
            kind = "sealed split" if best[r["group"]] in SEALED_SPLITS else "another split"
            r["split"], r["exclude_reason"] = "excluded", f"near-duplicate of {kind} ({best[r['group']]})"


def audit_near_duplicates(rows: list[dict], chunk: int = 2048) -> int:
    """Exhaustive near-duplicate check after assignment; returns how many rows it excluded.

    The LSH banding in group_duplicates only guarantees a comparison for pairs <= 3 bits apart
    (4 bands x 16 bits), so some 4-6-bit pairs are never grouped. Grouping stays as it is (changing it
    would reshuffle every split, sealed ones included); instead every missed pair that crosses splits
    loses its lower-priority member, so the sealed split keeps its image. Group ids are unchanged.
    """
    rank = {s: i for i, s in enumerate(PRIORITY)}
    idx = [i for i, r in enumerate(rows) if r["phash"] and r["split"] in rank]
    ph = np.array([int(rows[i]["phash"], 16) for i in idx], dtype=np.uint64)
    dh = np.array([int(rows[i]["dhash"], 16) for i in idx], dtype=np.uint64)
    excluded = 0
    for start in range(0, len(idx), chunk):
        near = (np.bitwise_count(ph[start:start + chunk, None] ^ ph[None, :]) <= NEAR_DUP_BITS) & \
               (np.bitwise_count(dh[start:start + chunk, None] ^ dh[None, :]) <= DHASH_CONFIRM_BITS)
        for a, b in zip(*np.nonzero(near)):
            ra, rb = rows[idx[start + a]], rows[idx[b]]
            if ra is rb or "excluded" in (ra["split"], rb["split"]):
                continue
            if ra["split"] == rb["split"]:
                # Same split is harmless for split membership, but inside the CV pools a missed pair with
                # two group ids could land in opposite folds. None exist today; fail loudly if one appears.
                if ra["split"] in FOLDED_SPLITS and ra["group"] != rb["group"]:
                    raise RuntimeError(f"near-duplicates in {ra['split']} with different groups: "
                                       f"{ra['path']} / {rb['path']}; merge their groups before folding")
                continue
            loser = ra if rank[ra["split"]] > rank[rb["split"]] else rb
            winner = rb if loser is ra else ra
            loser["exclude_reason"] = f"near-duplicate (exhaustive audit) of {winner['split']}"
            loser["split"] = "excluded"
            excluded += 1
    return excluded


def main(root: Path, out: Path) -> None:
    rows = collect(root)
    with Pool() as p:
        hashes = p.map(hash_image, [r["path"] for r in rows], chunksize=64)
    for r, (s, ph, dh) in zip(rows, hashes):
        r["sha256"], r["phash"], r["dhash"] = s, ph, dh
    groups = group_duplicates([r["sha256"] for r in rows], [r["phash"] for r in rows], [r["dhash"] for r in rows])
    for r, g in zip(rows, groups):
        r["group"] = g
    assign(rows)
    print(f"exhaustive near-duplicate audit excluded {audit_near_duplicates(rows)} rows")

    fields = ["path", "source", "source_split", "source_label", "canonical", "split", "group",
              "sha256", "phash", "dhash", "exclude_reason"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out.parent / "sealed_test.sha256").write_text(
        "".join(f"{r['split']}  {r['sha256']}  {r['path']}\n" for r in rows if r["split"] in SEALED_SPLITS))
    conflicts = sum(1 for labels in _labels_by_group(rows).values() if len(labels) > 1)

    print(f"images: {len(rows)}  dup groups with >1 image: "
          f"{sum(1 for c in Counter(r['group'] for r in rows).values() if c > 1)}")
    print(f"duplicate groups with conflicting labels: {conflicts}")
    print("excluded:", dict(Counter(r["exclude_reason"] for r in rows if r["split"] == "excluded")))
    table = Counter((r["canonical"], r["split"]) for r in rows if is_supported(r))
    splits = ["pv_train", "pv_val", "pv_calib", "field_dev", SEALED]
    print(f"{'class':32}" + "".join(f"{s:>21}" for s in splits))
    for c in sorted({k[0] for k in table}):
        print(f"{c:32}" + "".join(f"{table[(c, s)]:>21}" for s in splits))
    print("ood:", dict(Counter((r["source"], r["split"]) for r in rows if is_ood(r))))
    print("other crops:", dict(Counter(r["split"] for r in rows if r["canonical"] == "OTHER_CROP")))


def _labels_by_group(rows: list[dict]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        out[r["group"]].add(r["canonical"])
    return out


if __name__ == "__main__":
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    main(data_root, Path("data/manifest.csv"))
