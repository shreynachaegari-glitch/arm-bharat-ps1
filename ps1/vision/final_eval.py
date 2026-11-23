"""The sealed evaluation: deployed .pte + app preprocessing + quality gate + bound decision config.

One pinned configuration, ever. The real run needs PS1_FINAL_EVAL=1 and gates Sreyesh has confirmed. Everything
fixable is checked before the lock (gates, model, class order, set sizes, image hashes against the listing, and
that every image decodes; no model output is produced). The lock then pins the hashes of the model, decision
config, gates, manifest and quality thresholds, at one fixed path whatever --out says. After a crash, the same
pins may resume, because identical inputs can only reproduce identical numbers; different pins are refused.
Scores go to disk image by image, so a crash loses nothing. No retuning on these images afterwards, whatever the result.

Not exercised here: the input gate (W6). `built` is refused until this harness runs it; `waived` is Sreyesh's call
and means the sealed numbers exclude it (false accepts are then an upper bound, coverage is optimistic).
--dry-run-dev runs the same code on dev splits. v6_final trained on those, so it checks mechanics, not performance.
Usage: python -m ps1.vision.final_eval --pte <m>.pte --decision <d>.json --gates scripts/final/gates.json --out runs/final_eval
"""
import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ps1 import quality_gate
from ps1.data.build_splits import OTHER_CLASS, SEALED_SPLITS
from ps1.decision import DecisionConfig
from ps1.vision.dataset import MANIFEST, REPO_ROOT, read_manifest, select
from ps1.vision.gate_sheet import METRICS, SEALED_N, sheet_row
from ps1.vision.runtime import VisionModel, load_vision

LOCK_PATH = REPO_ROOT / "data/SEALED_RUN_STARTED.json"     # fixed: a different --out must not give a second run
SEALED_HASHES = REPO_ROOT / "data/sealed_test.sha256"
ALLOWED_INPUT_GATE = {"waived"}                            # add "built" once _evaluate runs the input gate
SEALED_SETS = {"known": ("final_external_test", "plantdoc"), "seen": ("ood_test", "plantdoc"),
               "novel": ("ood_test", "maharashtra")}
DEV_SETS = {"known": ("field_dev", "plantdoc"), "seen": ("ood_dev", "plantdoc"), "novel": ("ood_dev", "maharashtra")}


def problems_before_sealed_run(gates: dict, cfg: DecisionConfig, pte_sha: str, quality_version: str,
                               decision_sha: str, model_classes: tuple[str, ...]) -> list[str]:
    """Every reason the run must not start, from the configuration alone; an empty list means go on."""
    out = []
    if gates.get("status") != "confirmed":
        out.append(f"gates status is {gates.get('status')!r}, not confirmed")
    if not gates.get("confirmed_by"):
        out.append("confirmed_by is empty")
    if gates.get("pte_sha256") != pte_sha or cfg.pte_sha256 != pte_sha:
        out.append("gates, decision config and .pte do not share one sha256")
    if gates.get("decision_sha256") != decision_sha:
        out.append("decision file sha256 differs from the one in the gates (temperature or threshold changed?)")
    if tuple(model_classes) != tuple(cfg.classes):
        out.append("class order differs between the model and the decision config")
    if gates.get("threshold") != cfg.threshold:
        out.append(f"gates threshold {gates.get('threshold')} differs from decision config {cfg.threshold}")
    if gates.get("quality_thresholds_version") != quality_version:
        out.append(f"quality thresholds in code are {quality_version}, gates say {gates.get('quality_thresholds_version')}")
    if "placeholder" in quality_version:
        out.append(f"quality thresholds {quality_version} are still a placeholder (W3)")
    if gates.get("input_gate") == "built":
        out.append("input gate is built, but this harness does not run it yet: wire it into the evaluation first")
    elif gates.get("input_gate") not in ALLOWED_INPUT_GATE:
        out.append(f"input gate is {gates.get('input_gate')!r} (W6): must be waived, or built and wired in")
    unknown = [m for part in ("blocking", "report_only") for m in gates.get(part, {}) if m not in METRICS]
    if unknown:
        out.append(f"gates name metrics that do not exist: {unknown}")
    return out


def verify_sealed_hashes(path: Path = SEALED_HASHES, root: Path = REPO_ROOT) -> dict[str, str]:
    """`<split>  <sha256>  <relative path>` lines from the split builder -> {absolute path: sha256}. Bytes only."""
    listed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        _split, sha, rel = line.split(maxsplit=2)
        image = (root / rel).resolve()
        if hashlib.sha256(image.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"sealed image changed since the split was built: {rel}")
        listed[str(image)] = sha
    if not listed:
        raise SystemExit(f"{path} is empty: it verifies nothing")
    return listed


def listing_problems(listed: dict[str, str], rows: list[dict], manifest_sealed_count: int) -> list[str]:
    """The images about to be scored must be the ones sealed before training (a manifest edit would show here)."""
    out = []
    if len(listed) != manifest_sealed_count:
        out.append(f"listing has {len(listed)} images, the manifest's sealed splits have {manifest_sealed_count}")
    for r in rows:
        if r["path"] not in listed:
            out.append(f"not in the sealed listing: {r['path']}")
        elif listed[r["path"]] != r["sha256"]:
            out.append(f"manifest hash differs from the listing: {r['path']}")
    return out


def size_problems(rows_by_set: dict[str, list], expected: dict[str, int]) -> list[str]:
    return [f"set {name} has {len(rows_by_set[name])} images, expected {want}" for name, want in expected.items()
            if len(rows_by_set[name]) != want]


def claim_lock(lock: Path, pins: dict) -> bool:
    """False = first claim. True = same pins, resuming. Different pins exit. The exclusive create is atomic."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(lock, "x") as f:
            json.dump({"pins": pins, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
        return False
    except FileExistsError:
        recorded = json.loads(lock.read_text())["pins"]
    if recorded != pins:
        changed = sorted(k for k in pins if recorded.get(k) != pins[k])
        raise SystemExit(f"sealed evaluation already started with another configuration ({lock}); changed: {changed}. "
                         "It cannot be repeated.")
    return True


def prepare_and_lock(lock: Path, pins: dict, rows_by_set: dict[str, list[dict]], expected: dict[str, int], listing: Path,
                     root: Path, manifest_sealed_count: int, decode: Callable[[dict], object]) -> bool:
    """All fixable problems surface here, before the single-use lock is touched."""
    problems = size_problems(rows_by_set, expected)
    rows = [r for part in rows_by_set.values() for r in part]
    if not problems:
        problems = listing_problems(verify_sealed_hashes(listing, root), rows, manifest_sealed_count)
    if problems:
        raise SystemExit("sealed run refused:\n- " + "\n- ".join(problems[:20]))
    for r in rows:
        try:
            decode(r)
        except Exception as exc:
            raise SystemExit(f"sealed run refused, image will not decode: {r['path']}: {exc}") from exc
    return claim_lock(lock, pins)


def score_rows(vision: VisionModel, rows: list[dict], quality: Callable, scores: Path | None = None) -> list[dict]:
    """As in the app, a photo the quality gate rejects stops before the model. Appends each record as it is made."""
    done: dict[str, dict] = {}
    if scores is not None and scores.exists():
        for line in scores.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:      # the last line of a killed run
                continue
            done[record["path"]] = record
    # Rewrite cleanly first: appending after a half-written line would corrupt the next record.
    if scores is not None:
        scores.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    for r in rows:
        if r["path"] in done:
            continue
        ok = bool(quality(r["path"]).ok)
        logits = [float(x) for x in vision.logits(vision.preprocess(r["path"]))] if ok else None
        done[r["path"]] = {"path": r["path"], "quality_ok": ok, "logits": logits}
        if scores is not None:
            with open(scores, "a") as f:
                f.write(json.dumps(done[r["path"]]) + "\n")
    return [done[r["path"]] for r in rows]


def apply_quality(probs: np.ndarray, passed: np.ndarray, other_idx: int) -> np.ndarray:
    """A photo the quality gate rejects never reaches the model in the app, so it can never be accepted."""
    out = probs.copy()
    out[~passed] = 0.0
    out[~passed, other_idx] = 1.0
    return out


def to_probs(records: list[dict], cfg: DecisionConfig) -> np.ndarray:
    n_classes = len(cfg.classes)
    logits = np.array([r["logits"] if r["quality_ok"] else [0.0] * n_classes for r in records], dtype=np.float64)
    z = logits / cfg.temperature
    z -= z.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return apply_quality(probs, np.array([r["quality_ok"] for r in records]), list(cfg.classes).index(OTHER_CLASS))


def evaluate_gates(row: dict, gates: dict) -> dict:
    """A gate with no estimate (nothing accepted) fails: an empty answer must not pass an error limit."""
    def one(metric: str, rule: dict) -> dict:
        r = row[metric]
        rate = r["rate"]
        if rate is None:
            passed = False
        elif "max" in rule:
            passed = rate <= rule["max"]
        else:
            passed = rate >= rule["min"]
        return {"rule": rule, "rate": rate, "ci95": [r["low"], r["high"]], "passed": passed}

    blocking = {m: one(m, rule) for m, rule in gates.get("blocking", {}).items()}
    report = {m: one(m, rule) for m, rule in gates.get("report_only", {}).items()}
    return {"blocking": blocking, "report_only": report, "all_blocking_passed": all(g["passed"] for g in blocking.values())}


def rows_for_sets(manifest: list[dict], sets: dict, classes: list[str]) -> dict[str, list[dict]]:
    known_classes = [c for c in classes if c != OTHER_CLASS]
    out = {}
    for name, (split, source) in sets.items():
        wanted = known_classes if name == "known" else None
        out[name] = [r for r in select(manifest, {split}, wanted) if r["source"] == source]
    return out


def _evaluate(vision: VisionModel, cfg: DecisionConfig, rows_by_set: dict[str, list[dict]], scores: Path | None) -> dict:
    classes = list(cfg.classes)
    records = {name: score_rows(vision, rows, quality_gate.check, scores) for name, rows in rows_by_set.items()}
    data = {"known": (to_probs(records["known"], cfg), np.array([classes.index(r["canonical"]) for r in rows_by_set["known"]])),
            "seen": to_probs(records["seen"], cfg), "novel": to_probs(records["novel"], cfg)}
    return {**sheet_row(data, classes, cfg.threshold),
            "quality_rejected": {name: sum(not r["quality_ok"] for r in recs) for name, recs in records.items()}}


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--gates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run-dev", action="store_true", help="same code on non-sealed dev splits; mechanics only")
    args = ap.parse_args()

    vision, cfg = load_vision(args.pte), DecisionConfig.load(args.decision)
    gates = json.loads(Path(args.gates).read_text())
    version = quality_gate.QualityThresholds().version
    decision_sha = _sha(args.decision)
    problems = problems_before_sealed_run(gates, cfg, vision.pte_sha256, version, decision_sha, vision.classes)
    pins = {"pte_sha256": vision.pte_sha256, "decision_sha256": decision_sha, "gates_sha256": _sha(args.gates),
            "manifest_sha256": _sha(MANIFEST), "quality_thresholds": version}
    info = {"pins": pins, "decision": cfg.version, "threshold": cfg.threshold, "temperature": cfg.temperature}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(taxonomy=vision.taxonomy)

    if args.dry_run_dev:
        info["note"] = "DRY RUN on dev splits the model trained on: mechanics only, not a performance result"
        info["sealed_run_would_be_blocked_by"] = problems
        sets, name, scores = DEV_SETS, "dry_run_dev.json", None
        rows_by_set = rows_for_sets(manifest, sets, list(cfg.classes))
    else:
        if os.environ.get("PS1_FINAL_EVAL") != "1":
            raise SystemExit("set PS1_FINAL_EVAL=1 for the single final run")
        if problems:
            raise SystemExit("sealed run refused:\n- " + "\n- ".join(problems))
        sets, name = SEALED_SETS, "final_eval.json"
        rows_by_set = rows_for_sets(manifest, sets, list(cfg.classes))
        sealed_count = sum(r["split"] in SEALED_SPLITS for r in manifest)
        expected = {k: SEALED_N[k] for k in sets}
        resumed = prepare_and_lock(LOCK_PATH, pins, rows_by_set, expected, SEALED_HASHES, REPO_ROOT, sealed_count,
                                   lambda r: vision.preprocess(r["path"]) if quality_gate.check(r["path"]).ok else None)
        info["resumed"] = resumed
        scores = out / f"sealed_scores_{hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()[:12]}.jsonl"

    row = _evaluate(vision, cfg, rows_by_set, scores)
    result = {**info, "sets": {k: list(v) for k, v in sets.items()}, "row": row, "gates": evaluate_gates(row, gates),
              "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
