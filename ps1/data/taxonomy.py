"""Build taxonomy_map.csv from the *downloaded* folder labels (v4 2.1-2.2).

Counts come from the files on disk, never from assumptions. A mapping row
records the source, exact source label, canonical class, rationale and review
status, so a reviewer can accept or reject each one.

Usage: python -m ps1.data.taxonomy [data_root]
"""
import csv
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# (source, source_label) -> (canonical, rationale, review_status)
# review_status: "proposed" until Sreyesh/agri reviewer signs off; "needs_review" when the match is not exact.
MAPPING: dict[tuple[str, str], tuple[str, str, str]] = {
    ("plantvillage", "Tomato___Bacterial_spot"): ("tomato_bacterial_spot", "same disease name", "proposed"),
    ("plantvillage", "Tomato___Early_blight"): ("tomato_early_blight", "same disease name", "proposed"),
    ("plantvillage", "Tomato___Late_blight"): ("tomato_late_blight", "same disease name", "proposed"),
    ("plantvillage", "Tomato___Leaf_Mold"): ("tomato_leaf_mold", "same disease name", "proposed"),
    ("plantvillage", "Tomato___Septoria_leaf_spot"): ("tomato_septoria_leaf_spot", "same disease name", "proposed"),
    ("plantvillage", "Tomato___Tomato_mosaic_virus"): ("tomato_mosaic_virus", "same disease name", "proposed"),
    ("plantvillage", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"): ("tomato_yellow_leaf_curl_virus", "same disease name", "proposed"),
    ("plantvillage", "Tomato___healthy"): ("tomato_healthy", "healthy leaf", "proposed"),
    ("plantvillage", "Tomato___Spider_mites Two-spotted_spider_mite"): ("UNSUPPORTED", "pest; no PlantDoc counterpart; pests deferred (v4 sec 3)", "proposed"),
    ("plantvillage", "Tomato___Target_Spot"): ("UNSUPPORTED", "no PlantDoc counterpart; usable as unsupported-disease OOD", "proposed"),
    ("plantdoc", "Tomato two spotted spider mites leaf"): ("UNSUPPORTED", "pest; pests deferred (v4 sec 3)", "proposed"),
    ("plantdoc", "Tomato leaf bacterial spot"): ("tomato_bacterial_spot", "same disease name", "proposed"),
    ("plantdoc", "Tomato Early blight leaf"): ("tomato_early_blight", "same disease name", "proposed"),
    ("plantdoc", "Tomato leaf late blight"): ("tomato_late_blight", "same disease name", "proposed"),
    ("plantdoc", "Tomato mold leaf"): ("tomato_leaf_mold", "'mold' assumed = leaf mold (Passalora fulva); check images", "needs_review"),
    ("plantdoc", "Tomato Septoria leaf spot"): ("tomato_septoria_leaf_spot", "same disease name", "proposed"),
    ("plantdoc", "Tomato leaf mosaic virus"): ("tomato_mosaic_virus", "same disease name", "proposed"),
    ("plantdoc", "Tomato leaf yellow virus"): ("tomato_yellow_leaf_curl_virus", "'yellow virus' assumed = TYLCV; check images", "needs_review"),
    ("plantdoc", "Tomato leaf"): ("tomato_healthy", "PlantDoc's plain 'Tomato leaf' is its healthy class", "needs_review"),
}

# Tomato Leaf Dataset, Maharashtra (Mendeley rv3kxfv47y v1, CC BY 4.0; labelled by a Dr. BSKKV Dapoli team).
MAPPING.update({
    ("maharashtra", "Alternaria"): ("tomato_early_blight", "Alternaria solani = early blight", "proposed"),
    ("maharashtra", "Leaf miner"): ("UNSUPPORTED", "pest; Indian field OOD material", "proposed"),
    ("maharashtra", "Insect attack"): ("UNSUPPORTED", "pest; Indian field OOD material", "proposed"),
    ("maharashtra", "Physiological disorder"): ("UNSUPPORTED", "non-disease disorder; Indian field OOD", "proposed"),
    ("maharashtra", "Alternaria+Leafminer"): ("MIXED", "two problems in one image: excluded", "proposed"),
})

LICENCES = {"maharashtra": "CC-BY-4.0 (Mendeley rv3kxfv47y)","plantvillage": "CC-BY-SA-3.0 (declared in repo README.md front matter)",
            "plantdoc": "CC-BY-4.0"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def source_folders(root: Path) -> list[tuple[str, str, str, Path]]:
    """(source, split, label, path) for every class folder on disk."""
    rows = []
    for label_dir in sorted((root / "plantvillage/raw/color").glob("*/")):
        rows.append(("plantvillage", "all", label_dir.name, label_dir))
    for split in ("train", "test"):
        for label_dir in sorted((root / "plantdoc" / split).glob("*/")):
            rows.append(("plantdoc", split, label_dir.name, label_dir))
    for label_dir in sorted((root / "maharashtra/Real world data").glob("*/")):
        rows.append(("maharashtra", "all", label_dir.name, label_dir))
    return rows


def build(root: Path, out_csv: Path) -> list[dict]:
    rows = []
    for source, split, label, path in source_folders(root):
        canonical, rationale, status = MAPPING.get(
            (source, label), ("OTHER_CROP" if not label.lower().startswith("tomato") else "UNMAPPED",
                              "non-tomato folder: input-gate 'other_plant' material", "n/a"))
        rows.append(dict(source=source, split=split, source_label=label, canonical=canonical,
                         count=count_images(path), rationale=rationale, review_status=status,
                         licence=LICENCES[source]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def coverage(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Per canonical tomato class: PlantVillage count and PlantDoc train/test counts."""
    table: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r["canonical"].startswith("tomato_"):
            continue
        key = {"plantvillage": "pv", "plantdoc": f"pd_{r['split']}"}.get(r["source"], "ext")
        table.setdefault(r["canonical"], {"pv": 0, "pd_train": 0, "pd_test": 0, "ext": 0})[key] += r["count"]
    return table


if __name__ == "__main__":
    data_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    built = build(data_root, Path("data/taxonomy_map.csv"))
    unmapped = [r for r in built if r["canonical"] == "UNMAPPED"]
    print(f"{'class':32} {'PV':>6} {'PD-train':>9} {'PD-test':>8} {'ext':>5}")
    for name, c in sorted(coverage(built).items()):
        print(f"{name:32} {c['pv']:>6} {c['pd_train']:>9} {c['pd_test']:>8} {c['ext']:>5}")
    if unmapped:
        print("UNMAPPED tomato folders:", [r["source_label"] for r in unmapped])
