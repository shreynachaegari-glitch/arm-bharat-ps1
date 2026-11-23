"""Download a public Mendeley Data dataset file by file, verifying each file's published sha256.

Folders become directories (they carry the class labels). A provenance TSV (url, sha256, path) is written next to
the files, so a later build can check nothing changed. Used 21 Nov for the extra field-data sources (CC BY 4.0).

  python scripts/fetch_mendeley.py --id jttrv2w27r --version 2 --out data/raw/bd_diu_2025
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.mendeley.com/public-api/datasets"
UA = {"User-Agent": "Mozilla/5.0 (dataset fetch)"}
WORKERS = 6                          # parallel downloads; modest, to be gentle with the public API
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".txt", ".csv", ".pdf", ".yaml", ".md"}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def safe_part(name: str) -> str:
    """Folder/file names come from the internet: no path separators or parent references."""
    clean = name.replace("/", "_").replace("\\", "_").strip()
    if clean in ("", ".", ".."):
        raise ValueError(f"unsafe name {name!r}")
    return clean


def folder_paths(ds: str, version: int) -> dict[str, Path]:
    folders = get_json(f"{API}/{ds}/folders/{version}")
    by_id = {f["id"]: f for f in folders}

    def path(fid: str) -> Path:
        f = by_id[fid]
        parent = f.get("parent_id")
        return (path(parent) if parent in by_id else Path()) / safe_part(f["name"])
    return {fid: path(fid) for fid in by_id}


def download(url: str, dest: Path, sha: str) -> None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("sha256 mismatch")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return
        except Exception as e:               # retried, then reported with the file name
            if attempt == 2:
                raise RuntimeError(f"{dest.name}: {e}") from e
            time.sleep(2 * (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = get_json(f"{API}/{args.id}")
    licence = (meta.get("data_licence") or {}).get("short_name")
    if licence != "CC BY 4.0":
        print(f"licence is {licence!r}, not CC BY 4.0: stopping", file=sys.stderr)
        return 1
    folders = folder_paths(args.id, args.version)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, skipped, todo = [], [], []
    for fid in [None, *folders]:
        q = f"version={args.version}" + (f"&folder_id={fid}" if fid else "&folder_id=root")
        for f in get_json(f"{API}/{args.id}/files?{q}"):
            cd = f["content_details"]
            rel = (folders[fid] if fid else Path()) / safe_part(f["filename"])
            if rel.suffix.lower() not in ALLOWED_SUFFIXES:
                skipped.append(str(rel))
                continue
            dest = args.out / rel
            if not (dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == cd["sha256_hash"]):
                todo.append((cd["download_url"], dest, cd["sha256_hash"]))
            rows.append(f"{cd['download_url']}\t{cd['sha256_hash']}\t{rel}")
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda t: download(*t), todo))     # list() re-raises the first failure
    (args.out / "PROVENANCE.tsv").write_text(
        f"# Mendeley {args.id} v{args.version}, {licence}, {meta.get('name', '').strip()}, fetched {time.strftime('%Y-%m-%d %H:%M')}\n"
        + "\n".join(rows) + "\n")
    print(f"{args.id}: {len(rows)} files verified into {args.out}; skipped {len(skipped)} (suffix): {skipped[:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
