"""Build one Markdown file with all PS1 results (read from run artifacts) and the complete code."""
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps1.data.build_splits import SUPPORTED  # noqa: E402  (single source of the class list)

DEPLOY_RECIPE_RUN = "v5b"   # recipe of the deploy candidates (v6_final 31 Oct; v7_final = same + EMA, 21 Nov)
DEPLOY_RUN = "v7_final"     # Option A (21 Nov): chosen by the pre-registered rule in runs/option_a_summary.json
OUT = Path(os.environ.get("PS1_REPORT_OUT", ROOT / "build/code_and_results.md"))
LANG = {".py": "python", ".sh": "bash", ".yaml": "yaml", ".toml": "toml", ".txt": "text", "": "text"}


def short(c: str) -> str:
    return c.replace("tomato_", "")


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def curve(log: Path) -> list[dict]:
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.startswith("{")]
    return [r for r in rows if not r.get("ema")]          # the EMA summary line is not an epoch


def section_env() -> str:
    lock = (ROOT / "requirements.lock").read_text().splitlines()
    pins = [l for l in lock if l.split("==")[0] in ("torch", "torchvision", "executorch", "numpy", "pydantic",
                                                     "scikit-learn", "opencv-python-headless", "imagehash", "setuptools")]
    proc = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"], cwd=ROOT,
                          capture_output=True, text=True)
    tests = proc.stdout.strip().splitlines()[-1]
    if proc.returncode != 0:
        raise SystemExit(f"tests failed, report not written: {tests}")
    return "\n".join([
        "## 1. Environment",
        "- Host: Rocky Linux 8.10 (glibc 2.28), untouched; 20 cores, 31 GB RAM, no GPU. All Python in a user-space `uv` venv (Python 3.12).",
        "- C++ ExecuTorch runtime built in podman `python:3.11-bookworm` (glibc 2.36), image `localhost/ps1-et:0.6.0`.",
        "- Pinned stack (from `requirements.lock`): " + ", ".join(f"`{p}`" for p in pins),
        f"- Test suite at generation time: **{tests}**",
        "- Known gap: `requirements.lock` was frozen on 30 Oct before `py7zr` (used once to unpack the Maharashtra .7z) was installed; regenerate the lock before committing.",
    ])


def section_g1() -> str:
    log = (ROOT / "logs/g1_smoke.log").read_text()
    s = json.loads(re.search(r"PyTorchObserver (\{.*\})", log).group(1))
    rows = [["Model", "`executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` @ `911221e0`"],
            [".pte sha256", "`8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b` (matches HF LFS oid)"],
            ["tokenizer sha256", "`82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55`"],
            ["Runtime", "ExecuTorch v0.6.0 (`e67ef3b1`), the release the .pte was exported with"],
            ["Prompt / generated tokens", f"{s['prompt_tokens']} / {s['generated_tokens']}"],
            ["Model load", f"{(s['model_load_end_ms'] - s['model_load_start_ms']) / 1000:.2f} s"],
            ["Time to first token", f"{(s['first_token_ms'] - s['inference_start_ms']) / 1000:.2f} s"],
            ["Decode speed", f"{(s['generated_tokens'] - 1) / ((s['inference_end_ms'] - s['first_token_ms']) / 1000):.1f} tok/s (4 threads)"]]
    return "\n".join(["## 2. G1: Llama runtime on x86 (functional pass)",
                      "One offline run (`--network=none`), with vision training running in parallel, so timings are pessimistic. The 10-run benchmark is still pending.",
                      "", table(["Item", "Value"], rows),
                      "", "Wrapper check: runner logs go to stderr; generated text passed the checker and was revealed; 7.6 s end-to-end incl. container start and cold load."])


def section_data() -> str:
    rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
    splits = ["pv_train", "pv_val", "pv_calib", "ext_train", "field_dev", "final_external_test"]
    c = Counter((r["canonical"], r["split"]) for r in rows)
    sup = [[short(k)] + [c[(k, s)] for s in splits] for k in SUPPORTED]
    sup.append(["**total**"] + [sum(c[(k, s)] for k in SUPPORTED) for s in splits])
    ood = Counter((r["source"], r["split"]) for r in rows if r["split"].startswith("ood"))
    other = Counter(r["split"] for r in rows if r["canonical"] == "OTHER_CROP")
    def reason(r: dict) -> str:
        why = r["exclude_reason"]
        if not why:
            return "mixed label (Maharashtra Alternaria+Leafminer)" if r["canonical"] == "MIXED" else "unassigned"
        if "exhaustive audit" in why:
            return "near-duplicate found by exhaustive audit"
        return re.sub(r" \(.*\)$", "", why)
    excl = Counter(reason(r) for r in rows if r["split"] == "excluded")
    sha = hashlib.sha256((ROOT / "data/manifest.csv").read_bytes()).hexdigest()
    return "\n".join([
        "## 3. Data",
        "Sources: PlantVillage `spMohanty/PlantVillage-Dataset@7f7ecc7e` (CC-BY-SA-3.0); PlantDoc `pratikkayal/PlantDoc-Dataset@5467f601` (CC-BY-4.0); Maharashtra tomato set, Mendeley `rv3kxfv47y` v1 (CC BY 4.0, BSKKV Dapoli team). GTLD (only 7 sample images published) and the Bangladesh set (lesion-level boxes, unrecoverable class map) were rejected; Tomato-Village has no licence.",
        f"Images in manifest: **{len(rows):,}**; manifest sha256 `{sha[:16]}…`. Sealed splits (`final_external_test`, `ood_test`, `other_test`) are refused by the loader unless `PS1_FINAL_EVAL=1`.",
        "", "**Supported classes per split** (shipped: 4 diseases + healthy, Sreyesh 30 Oct):", "",
        table(["class"] + splits, sup), "",
        "**OOD (unsupported problems) per source:** " + ", ".join(f"{s}/{sp}: {n}" for (s, sp), n in sorted(ood.items())),
        "", "**Other crops (input-gate material):** " + ", ".join(f"{k}: {v}" for k, v in sorted(other.items())),
        "", "**Excluded images:** " + ", ".join(f"{k}: {v}" for k, v in excl.most_common()),
        "", "Counts are from the current (post-audit, 31 Oct) manifest. v0-v3 were trained on the pre-audit manifest, which differed only by the 187 audit exclusions (e.g. pv_train 5,868 then vs 5,865 now; ood_train 4,879 vs 4,868); `final_external_test` and `field_dev` are identical in both.",
        "", "Dedup: exact sha256 + pHash (≤6 bits, LSH banding) confirmed by dHash (≤10 bits); then an exhaustive near-duplicate audit (31 Oct) that removed 187 cross-split copies the LSH missed, without changing groups or the sealed/field splits.",
    ])


def section_training() -> str:
    parts = ["## 4. Classifier training", "All selection on held-out field data only; lab `pv_val` is near-saturated and uninformative."]
    for name, note in [("v0", "MobileNetV3-Small, PlantVillage only"), ("v1", "+ 70% field-background compositing, backbone LR x0.1")]:
        h = curve(ROOT / f"runs/{name}/train.log")
        parts += ["", f"**{name}** ({note}; stopped early once field F1 kept falling):", "",
                  table(["epoch", "lab val F1", "field_dev F1"], [[e["epoch"], f3(e["pv_val_f1"]), f3(e.get("field_dev_f1"))] for e in h])]
    notes = {"v2": "EfficientNet-B0 + one field fold x8 in training, selected on the other fold",
             "v3": "v2 + 'other tomato problem' class (outlier exposure)",
             "v4": "t2 taxonomy (bacterial spot + Septoria merged into leaf spot), stratified folds, fixed 10 epochs (last.pt)",
             "v5m": "MobileNetV3-Small, v5b recipe (baseline)",
             **{f"v7s{s}": f"Option A: v5b recipe + EMA 0.998 from epoch 4, seed {s}; rows are last.pt, EMA in section 4d" for s in (0, 1, 2)}}
    for name in sorted({p.parent.name.rsplit("_f", 1)[0] for p in ROOT.glob("runs/v*_f0/train.log")}):
        h0, h1 = curve(ROOT / f"runs/{name}_f0/train.log"), curve(ROOT / f"runs/{name}_f1/train.log")
        if not h0 or not h1:
            continue
        parts += ["", f"**{name}** ({notes.get(name, '')}; F1 over classes present in each set):", "",
                  table(["epoch", "lab val F1 f0", "held-out field F1 f0", "lab val F1 f1", "held-out field F1 f1"],
                        [[a["epoch"], f3(a["pv_val_f1"]), f3(a["field_f1"]), f3(b["pv_val_f1"]), f3(b["field_f1"])] for a, b in zip(h0, h1)])]
    return "\n".join(parts)


def section_probes() -> str:
    rows = []
    for bb in ("mnv3s_imagenet", "effnet_b0_imagenet", "convnext_tiny_imagenet"):
        p = json.loads((ROOT / f"runs/probe/probe_{bb}.json").read_text())
        rows.append([bb, f3(p["A_lab_only_field_f1"]), f3(p["B_with_field_folds_f1"])])
    return "\n".join(["## 5. Frozen-backbone probes (triage)",
                      "Logistic regression on frozen features; field macro-F1. A = lab data only; B = + one field fold (2-fold mean).",
                      "", table(["backbone", "A lab only", "B + field fold"], rows),
                      "", "Field images in training are the main lever. DINOv2 via `torch.hub` was blocked by the auto-mode safety check (runs code fetched from GitHub); `timm` weights-only route pending approval."])


def _oe_table(path: Path) -> list[str]:
    r = json.loads(path.read_text())
    n = r["n"]
    head = (f"**{path.parent.name}** (taxonomy {r.get('taxonomy', 't1')}, folds {r.get('fold_scheme', 'roundrobin')}, "
            f"epochs {r.get('epochs', '?')}; T={r['temperature']:.3f}; n known {n['known_field']}, seen-type OOD "
            f"{n['ood_field_seen_types']}, novel Indian OOD {n['ood_novel_india']}). Known-class field macro-F1 "
            f"{r['known_field_macro_f1']:.3f}; known sent to 'other' {r['known_sent_to_other']:.3f}.")
    rows = [[p["threshold"], f3(p["known_coverage"]), f3(p["accepted_error"]), f3(p["false_accept_ood_field_seen_types"]),
             f3(p["false_accept_ood_novel_india"])] for p in r["points"]]
    return ["", head, "", table(["t", "field coverage", "accepted error", "FA seen-type field OOD", "FA novel Indian OOD"], rows)]


def section_abstention() -> str:
    v2 = json.loads((ROOT / "runs/v2_final/abstention_dev_maharashtra_plantdoc.json").read_text())
    r2 = [[m, f3(d["auroc_known_vs_ood"])] + [f"{p['known_coverage']:.2f} / {f3(p['accepted_error'])}" for p in d["operating_points"]]
          for m, d in v2["methods"].items()]
    parts = ["## 6. Abstention", "Dev estimates: temperature and thresholds are fit and scored on the same out-of-fold data.", "",
             f"**v2, confidence only** (field OOD = PlantDoc + Maharashtra; T={v2['temperature']:.3f}). Cells: field coverage / accepted error at OOD false-accept 5 / 10 / 20 / 30%:", "",
             table(["score", "AUROC", "FA 5%", "FA 10%", "FA 20%", "FA 30%"], r2)]
    for path in sorted(ROOT.glob("runs/v*/abstention_oe_dev.json")):
        parts += _oe_table(path)
    ver = json.loads((ROOT / "runs/v3/verification_2026-10-31.json").read_text())
    lfl = ver["like_for_like"]["rows"]
    parts += ["", f"**Like-for-like v2 vs v3** ({ver['source']}):", "",
              table(["coverage", "v2 FA", "v3 FA", "v2 accepted error", "v3 accepted error"],
                    [[f"{r['coverage']:.2f}", f3(r["v2_false_accept"]), f3(r["v3_false_accept"]), f3(r["v2_accepted_error"]),
                      f3(r["v3_accepted_error"])] for r in lfl])]
    bs = ver["bootstrap"]
    parts += ["", f"Bootstrap ({bs['resamples']} resamples, n={bs['n']}): v2 {bs['v2']['f1']:.3f} {bs['v2']['ci']}, "
              f"v3 {bs['v3']['f1']:.3f} {bs['v3']['ci']}, difference {bs['v3_minus_v2_ci']}."]
    cm = ver["v3_confusion"]
    parts += ["", "v3 out-of-fold confusion (rows true, columns predicted: " + ", ".join(cm["classes"]) + "): " +
              "; ".join(f"{c} {row}" for c, row in zip(cm["classes"], cm["rows_true_5"])) + "."]
    return "\n".join(parts)


def section_export() -> str:
    parts = ["## 7. Vision export parity (x86; Pi numbers come at G2)"]
    files = sorted(set(ROOT.glob("runs/*/export/*_parity.json")) | set(ROOT.glob("runs/*/export/export_parity.json")))
    for path in files:
        p = json.loads(path.read_text())
        sets = [k for k in ("pv_val", "field_heldout") if k in p]
        rows = [[k, p[k]["n"], f3(p[k]["fp32_macro_f1"]), f3(p[k]["pte_macro_f1"]), f"{p[k]['delta_pte_minus_fp32']:+.3f}",
                 f3(p[k].get("top1_agreement")), f"{p[k]['pte_median_ms_x86']:.1f}"] for k in sets]
        parts += ["", f"**{path.parent.parent.name} / {Path(p['pte']).name}** ({p.get('precision', 'int8')}, "
                  f"{p['pte_bytes']:,} B, sha256 `{p['pte_sha256'][:16]}...`, {p['calibration_images']} calibration images):", "",
                  table(["set", "n", "FP32", ".pte", "delta", "top-1 agreement", "x86 ms/img"], rows)]
    parts += ["", "Gate (draft): .pte vs FP32 macro-F1 delta >= -0.02. INT8 (v3) failed it; FP32 is the default export."]
    return "\n".join(parts)


def section_review() -> str:
    return "\n".join([
        "## 8. Review (31 Oct): verification, code review, MLE review", "",
        table(["#", "Finding", "Status"], [
            ["1", "pHash LSH only guaranteed comparison ≤3 bits → 364 missed near-dup pairs (197 cross-split; 12 ood_train↔sealed ood_test; none in final test/field_dev)", "fixed: exhaustive audit, 187 copies excluded"],
            ["2", "ood_train relabel after straddle check", "fixed (0 real instances)"],
            ["3", "checker blocked safe hedges ('cannot be confirmed')", "fixed, checker c2"],
            ["4", "truncated Llama output accepted", "fixed: complete-sentence rule"],
            ["5", "probe cache keyed by name only", "fixed: keyed by row hash"],
            ["6", "new sources could merge base groups", "open; verified per build (0 moves)"],
            ["7", "abstain crash on None AUROC", "fixed"],
            ["8", "dark shadows reported TOO_BRIGHT", "fixed (red→green test)"],
            ["9", "G1 script stripped prompt newlines; prompt.txt drifted", "fixed; prompt regenerated from build_prompt"],
            ["10", "SEALED_SPLITS defined twice", "fixed"],
            ["MLE", "best-epoch selection optimism (fold 0: 0.522 vs 0.506 last epoch)", "last.pt now saved for fixed-epoch reporting"],
            ["MLE", "unseeded composite RNG; no data/config in checkpoints", "fixed"],
            ["MLE", "train/serve skew risk; no .pte metadata; lab-only INT8 calibration", "fixed: normalisation in graph, sidecar JSON, field calibration"],
            ["claim", "card `cause` never reached Llama", "fixed"],
        ]), "",
        "**Draft promotion gates (to confirm before the sealed run):** INT8 Δ ≥ −0.02; sealed accepted error ≤ 0.30; seen-type field OOD false-accept ≤ 0.10; novel Indian OOD false-accept ≤ 0.30 (report); field coverage ≥ 0.25; Pi 5 Screen-1 p95 ≤ 1.5 s.",
        "", "**Pending decisions:** promotion gates; INT8 vs FP32 vision; merge bacterial spot + Septoria (agreed) then retrain on clean data; threshold (0.7 proposed); `timm`/DINOv2.",
    ])


def _at(points: list[dict], t: float) -> dict:
    return next(x for x in points if abs(x["threshold"] - t) < 1e-9)


def section_baseline() -> str:
    """21 Nov: MobileNetV3-Small vs EfficientNet-B0, same v5b recipe and folds (only --arch differs)."""
    arms = [("EfficientNet-B0 (deploy recipe)", "runs/v5b", "runs/v6_final/export/tomato_effnet_b0_t2_fp32_xnnpack_parity.json"),
            ("MobileNetV3-Small (baseline)", "runs/v5m", "runs/v5m_f0/export/tomato_mnv3s_t2_fp32_xnnpack_parity.json")]
    rows = []
    for name, run, parity in arms:
        r = json.loads((ROOT / run / "abstention_oe_dev.json").read_text())
        p = json.loads((ROOT / parity).read_text())
        a7, a8 = _at(r["points"], 0.7), _at(r["points"], 0.8)
        rows.append([name, f3(r["known_field_macro_f1"]), f"{a7['known_coverage']:.0%} / {a7['accepted_error']:.0%}",
                     f"{a8['known_coverage']:.0%} / {a8['accepted_error']:.0%}", f"{a7['false_accept_ood_novel_india']:.0%}",
                     f"{p['pte_bytes'] / 1e6:.1f}", f"{p['pv_val']['pte_median_ms_x86']:.1f}"])
    return "\n".join([
        "## 4b. Baseline: MobileNetV3-Small vs EfficientNet-B0 (21 Nov)",
        "Same v5b recipe, same two stratified folds, Maharashtra out of training; only `--arch` differs "
        "(`scripts/run_baseline_mnv3.sh`). Out-of-fold dev estimates; size and latency from the FP32 .pte "
        "(EfficientNet: v6_final export; MobileNetV3: fold-0 export), x86, not the Pi.", "",
        table(["model", "field macro-F1", "t=0.7 answers / error", "t=0.8 answers / error", "t=0.7 FA novel Indian",
               ".pte MB", "x86 ms/img"], rows), "",
        "Macro-F1 is the same within noise (n=263 known field photos). At the deploy thresholds EfficientNet-B0 answers "
        "more photos at a similar error, so it stays the deploy model; MobileNetV3-Small is 2.6x smaller and about 2x "
        "faster, but vision is milliseconds next to a seconds-long Llama call either way."])


def section_app() -> str:
    """21 Nov: offline web app + the real-model Screen 2 check (artifact written by scripts/llama_card_check.py)."""
    chk = json.loads((ROOT / "runs/llama_cards/card_check.json").read_text())
    rows = [[r["card"], "not called" if not r["called"] else r["shown_source"],
             "" if not r["called"] else r["checker_result"], "" if not r["called"] else "; ".join(r["reasons"]),
             "" if not r["called"] else f"{r['latency_s']:.1f}", "" if not r["called"] else r["shown_text"]]
            for r in chk["rows"]]
    shown = sum(1 for r in chk["rows"] if r.get("shown_source") == "llama")
    called = sum(1 for r in chk["rows"] if r["called"])
    return "\n".join([
        "## 4c. Offline app and Screen 2 (21 Nov)",
        "`ps1/app` (Flask + waitress, `scripts/run_app.sh`): photo in, Screen 1 (reviewed card) at once, Screen 2 (Llama "
        "rewrite of the reviewed text) only after the checker passes. Vision and Llama run on separate single-thread lanes; "
        "the photo is deleted once Screen 1 is built; the page loads nothing from the internet (CSP `default-src 'self'`). "
        "Screens: `docs/app-screens/`.", "",
        f"**Real Llama on every card** (`scripts/llama_card_check.py`, {chk['generated']}, checker {chk['checker']}, "
        f"greedy): {shown}/{called} disease cards show Llama text; the rest fall back to the reviewed text. Healthy has no "
        "actions, so Llama is not called.", "",
        table(["card", "shown", "checker", "reasons", "latency s (x86)", "text shown to the farmer"], rows), "",
        "Why rewrite mode: in free-explanation mode the 1B model added facts the card does not have (e.g. \"bacterial "
        "spot, which is also caused by Septoria\"; weather causes for late blight) and the lexical checker passed them. "
        "Guards added: card `cause_alternatives` (name every possible cause or none), required 'a photo cannot confirm' "
        "warning, refusal reject, preamble/quote clean-up that only removes text; the Advice validator re-runs them."])


def section_option_a() -> str:
    """21 Nov: v5b recipe x seeds 0-2, raw last.pt vs EMA, scored out-of-fold; arm picked by the pre-registered rule."""
    d = json.loads((ROOT / "runs/option_a_summary.json").read_text())

    def ms(s: dict, pct: bool) -> str:
        f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.3f}")
        return f"{f(s['mean'])} ± {f(s['sd'])} ({f(s['min'])}-{f(s['max'])})"
    rows = []
    for arm, name in (("raw", "raw last.pt"), ("ema", "EMA ema.pt")):
        a = d[arm]
        rows.append([name, ms(a["macro_f1"], False)] + [ms(a[t][k], True) for t in ("t0.7", "t0.78") for k in
                    ("known_coverage", "accepted_error", "false_accept_ood_field_seen_types", "false_accept_ood_novel_india")])
    rule = d["rule"]
    r_f, e_f = d["raw"]["macro_f1"], d["ema"]["macro_f1"]
    r_s, e_s = (d[x]["t0.78"]["false_accept_ood_field_seen_types"] for x in ("raw", "ema"))
    return "\n".join([
        "## 4d. Seeds and weight averaging (Option A, 21 Nov)",
        "v5b recipe, seeds 0/1/2 x the same two stratified folds; each run saves the last epoch (raw) and an exponential "
        "moving average of the weights (EMA, decay 0.998 from epoch 4, BatchNorm statistics included). Both are scored "
        "out-of-fold per seed (`scripts/run_option_a.sh`, `scripts/option_a_summary.py`). Cells: mean ± sd (min-max) over "
        "3 seeds. The seed spread is the honest error bar for every single-seed number in this report.", "",
        table(["arm", "field macro-F1", "t=0.7 answers", "t=0.7 error", "t=0.7 FA seen-type", "t=0.7 FA novel Indian",
               "t=0.78 answers", "t=0.78 error", "t=0.78 FA seen-type", "t=0.78 FA novel Indian"], rows), "",
        f"**Pre-registered rule** (written before any result): adopt EMA iff its mean F1 ≥ raw and its mean novel-Indian "
        f"false accepts at t=0.78 ≤ raw + 0.03. F1 not worse: {rule['f1_not_worse']}; novel FA within margin: "
        f"{rule['novel_fa_within_margin']} → **{rule['arm']}** ships as {DEPLOY_RUN}. EMA's main effect is stability "
        f"(F1 sd {r_f['sd']:.3f} → {e_f['sd']:.3f}; worst seed {r_f['min']:.3f} → {e_f['min']:.3f}), not a higher best "
        f"case: the best raw seed ({r_f['max']:.3f}) is the v5b single-run number. Not in the rule and reported as found: "
        f"mean seen-type false accepts at t=0.78 are {r_s['mean']:.1%} raw vs {e_s['mean']:.1%} EMA (seed sd about "
        f"{max(r_s['sd'], e_s['sd']):.1%})."])


def section_sealed() -> str:
    """21 Nov: the one sealed run (ps1.vision.final_eval), read from its result file; reported as is."""
    path = ROOT / "runs/final_eval/final_eval.json"
    if not path.is_file():
        return "## 4e. Sealed final test\nNot run yet."
    d = json.loads(path.read_text())
    gates = {**d["gates"]["blocking"], **d["gates"]["report_only"]}
    rows = []
    for name, m in d["row"].items():
        if not isinstance(m, dict) or "rate" not in m:
            continue
        g = gates.get(name)
        rule = "" if g is None else (f"≤ {g['rule']['max']}" if "max" in g["rule"] else f"≥ {g['rule']['min']}")
        kind = "blocking" if name in d["gates"]["blocking"] else "report only" if g else ""
        rows.append([name, f"{m['rate']:.3f} ({m['k']}/{m['n']})", f"{m['low']:.3f}-{m['high']:.3f}", rule, kind,
                     "" if g is None else ("pass" if g["passed"] else "FAIL")])
    pins = d["pins"]
    return "\n".join([
        "## 4e. Sealed final test (run once, 21 Nov)",
        f"`ps1.vision.final_eval` with `PS1_FINAL_EVAL=1`, finished {d['finished']}; decision {d['decision']} "
        f"(t={d['threshold']}, T={d['temperature']:.3f}); pins: .pte `{pins['pte_sha256'][:12]}`, decision "
        f"`{pins['decision_sha256'][:12]}`, gates `{pins['gates_sha256'][:12]}`, manifest `{pins['manifest_sha256'][:12]}`, "
        f"quality {pins['quality_thresholds']}; resumed: {d['resumed']}. Input gate waived. Quality-rejected: "
        + ", ".join(f"{k} {v}" for k, v in d["row"]["quality_rejected"].items()) + ".", "",
        table(["metric", "rate (k/n)", "95% interval", "gate", "kind", "result"], rows), "",
        f"**All blocking gates passed: {d['gates']['all_blocking_passed']}.** Reported as is; nothing was tuned after this run."])


def section_v8() -> str:
    """22 Nov: v8 = v7 recipe + extra Bangladesh field data (DIU), judged by the rule pre-registered on 21 Nov."""
    path = ROOT / "runs/v8_summary.json"
    if not path.is_file():
        return "## 4f. Extra field data (v8)\nNot run yet."
    s = json.loads(path.read_text())
    oof, neu = s["oof_t0.8"], s["neu_t0.8"]
    ex = {r: json.loads((ROOT / f"runs/{r}_final/extra_eval.json").read_text()) for r in ("v7", "v8")}
    india = {r: json.loads((ROOT / f"runs/{r}_final/india_check.json").read_text())["groups"] for r in ("v7", "v8")}

    def ms(m: dict) -> str:
        return f"{m['mean']:.3f} ± {m['sd']:.3f}"

    def kn(m: dict) -> str:
        return f"{m['k']}/{m['n']}"
    rows = [[r, ms(oof[r]["macro_f1"]), ms(oof[r]["accepted_error"]), ms(oof[r]["false_accept_ood_field_seen_types"]),
             f"{neu[r]['f1_forced']:.3f}", kn(neu[r]["raw"]["coverage"]), kn(neu[r]["raw"]["accepted_error"]),
             kn(neu[r]["raw"]["sick_called_healthy"]), kn(neu[r]["raw"]["false_accept_ood"]),
             kn({"k": india[r]["unsupported_unseen"]["false_accepts"], "n": india[r]["unsupported_unseen"]["n"]})]
            for r in ("v7", "v8")]
    conf = []
    for r in ("v7", "v8"):
        for true, preds in ex[r]["forced_choice_confusion"].items():
            conf.append([r, short(true)] + [preds.get(c, 0) for c in ex[r]["supported_classes"] + ["tomato_other_problem"]])
    cols = [short(c) for c in ex["v7"]["supported_classes"]] + ["other"]
    checks = ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in s["rule"].items() if k != "adopt_v8")
    return "\n".join([
        "## 4f. Extra field data (v8, after the sealed run)",
        "v7 recipe + 1,872 Bangladesh field images (DIU, Mendeley CC BY 4.0) for the supported classes and 3,522 for "
        "'other problem' (`data/manifest_extra.csv`; any duplicate of a main-manifest image, sealed included, removed). "
        "NEU (Bangladesh, Mendeley CC BY 4.0) held out: 1,488 known-class and 972 unsupported photos, never trained or "
        "selected on. Threshold 0.80 kept (not re-tuned). Rule written before the run (PS1 Extra Field Data v8 Plan).", "",
        table(["model", "PlantDoc OOF F1 (3 seeds)", "OOF error t=0.8", "OOF FA seen t=0.8", "NEU F1 forced",
               "NEU answered", "NEU wrong", "NEU sick→healthy", "NEU unsupported accepted", "India unsupported accepted"], rows), "",
        f"Rule checks: {checks}. **Rule result: adopt v8 = {s['rule']['adopt_v8']}.**", "",
        "NEU forced-choice confusion (rows true, columns predicted):", "",
        table(["model", "true"] + cols, conf), "",
        "Read with care: (1) most of the NEU gain at t=0.8 comes from v8 being less over-confident on this source "
        "(fewer answers, far fewer wrong ones); forced-choice F1 rose only modestly and v8 now mistakes NEU late blight "
        "for early blight, where v7 mistook NEU early blight for leaf spot. (2) DIU and NEU photos are both single "
        "detached leaves on white paper, so NEU is a new source but a familiar photo style for v8. (3) The PlantDoc "
        "out-of-fold F1 gain is inside the seed spread. (4) On the 34 Indian unsupported photos v8 accepted one, an "
        "'insect attack' leaf that looks nearly healthy, as healthy; v7 accepted none. "
        "**v7 stays the Phase 1 submitted model** (it carries the sealed result); v8 has no sealed test."])

def section_code() -> str:
    # Top-level asset folders only: a bare "data" match would also drop ps1/data/ (the split builder).
    skip_top = {".venv", "data", "artifacts", "runs", "logs"}
    text_suffixes = {".py", ".sh", ".yaml", ".toml", ".txt", ".lock"}

    def is_source(p: Path) -> bool:
        parts = p.relative_to(ROOT).parts
        if parts[0] in skip_top or any(x.startswith(".") and x != ".gitignore" for x in parts) or "__pycache__" in parts:
            return False
        return p.stat().st_size > 0 and (p.suffix in text_suffixes or p.name in ("Containerfile", ".gitignore"))

    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and is_source(p))
    order = lambda p: (0 if p.parts[len(ROOT.parts)] == "ps1" else 1 if p.parts[len(ROOT.parts)] == "scripts"
                       else 2 if p.parts[len(ROOT.parts)] == "cards" else 3 if p.parts[len(ROOT.parts)] == "tests" else 4, str(p))
    files.sort(key=order)
    total = sum(len(p.read_text().splitlines()) for p in files)
    out = ["## 9. Complete code", f"{len(files)} non-empty files, {total:,} lines, from `~/hello-world/arm-bharat-ps1/` (empty `__init__.py` files omitted).", "",
           "**Index**", ""]
    out += [f"- `{p.relative_to(ROOT)}` ({len(p.read_text().splitlines())} lines)" for p in files]
    for p in files:
        rel = p.relative_to(ROOT)
        lang = "dockerfile" if p.name == "Containerfile" else LANG.get(p.suffix, "text")
        out += ["", f"### `{rel}`", "", f"````{lang}", p.read_text().rstrip("\n"), "````"]
    return "\n".join(out)


def status_line() -> str:
    """Headline numbers of the newest outlier-exposure run, read from its artifact (t = 0.7 operating point)."""
    runs = [ROOT / "runs" / DEPLOY_RECIPE_RUN / "abstention_oe_dev.json"]   # not the newest name: v5m is a baseline
    r = json.loads(runs[-1].read_text())
    p = next(x for x in r["points"] if abs(x["threshold"] - 0.7) < 1e-9)
    return (f"**Status:** G1 passed on x86 (10/10 deterministic runs); data pinned and audited; deploy recipe {runs[-1].parent.name} "
            f"(taxonomy {r.get('taxonomy', 't1')}, {r.get('fold_scheme', 'roundrobin')} folds): single run (seed 0): out-of-fold "
            f"known-class field macro-F1 {r['known_field_macro_f1']:.3f}; at t=0.7 it answers {p['known_coverage']:.0%} of field photos with "
            f"{p['accepted_error']:.0%} error, lets {p['false_accept_ood_field_seen_types']:.0%} of seen-type unsupported "
            f"problems and {p['false_accept_ood_novel_india']:.0%} of novel Indian problems through (dev estimates). "
            f"{seed_line()}{final_candidate_line()}{sealed_line()}")


def sealed_line() -> str:
    path = ROOT / "runs/final_eval/final_eval.json"
    if not path.is_file():
        return "Sealed test never opened."
    d = json.loads(path.read_text())
    r = d["row"]
    verdict = "passed every blocking gate" if d["gates"]["all_blocking_passed"] else "FAILED a blocking gate"
    return (f"Sealed test (run once, {d['finished'][:16]}, t={d['threshold']}): {verdict}; answers "
            f"{r['coverage']['k']}/{r['coverage']['n']} with {r['accepted_error']['k']}/{r['accepted_error']['n']} wrong, "
            f"sick called healthy {r['sick_called_healthy']['k']}/{r['sick_called_healthy']['n']} (section 4e).")


def seed_line() -> str:
    path = ROOT / "runs/option_a_summary.json"
    if not path.is_file():
        return ""
    d = json.loads(path.read_text())
    arm = d["rule"]["arm"]
    f, p = d[arm]["macro_f1"], d[arm]["t0.7"]
    return (f"Over 3 seeds the shipped arm ({arm}) gives macro-F1 {f['mean']:.3f} ± {f['sd']:.3f} and at t=0.7 answers "
            f"{p['known_coverage']['mean']:.0%} with {p['accepted_error']['mean']:.0%} error (section 4d). ")


def final_candidate_line() -> str:
    """One sentence on the deploy candidate (trained on all non-sealed field data, so no fold estimates of its own)."""
    check, decision = ROOT / "runs" / DEPLOY_RUN / "india_check.json", ROOT / "runs" / DEPLOY_RUN / "export/decision.json"
    if not (check.is_file() and decision.is_file()):
        return ""
    c, d = json.loads(check.read_text())["groups"], json.loads(decision.read_text())
    eb, ood = c["early_blight_unseen"], c["unsupported_unseen"]
    return (f"Deploy candidate {DEPLOY_RUN} (v5b recipe + EMA on all non-sealed field data, seed 0; FP32 .pte sha {d['pte_sha256'][:12]}, "
            f"decision {d['version']}): on unseen Indian photos it accepts {ood['false_accepts']}/{ood['n']} unsupported "
            f"problems and gets {eb['correct_accepts']}/{eb['n']} early-blight photos right "
            f"({eb['n'] - eb['accepted']} abstained). ")


def main() -> None:
    head = "\n".join([
        "# PS1 Code and Results", "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} from the run artifacts in `~/hello-world/arm-bharat-ps1/runs/`, `logs/` and `data/manifest.csv` (tables are read from files, not retyped).", "",
        status_line(),
        "", "**Contents:** 1 Environment · 2 G1 Llama runtime · 3 Data · 4 Training · 4b Baseline · 4c App + Screen 2 · 4d Seeds + EMA · 4e Sealed test · 4f Extra data (v8) · 5 Probes · 6 Abstention · 7 INT8 export · 8 Review · 9 Complete code",
    ])
    body = [head, section_env(), section_g1(), section_data(), section_training(), section_baseline(), section_app(), section_option_a(), section_sealed(), section_v8(), section_probes(),
            section_abstention(), section_export(), section_review(), section_code()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(body) + "\n")
    print(OUT, f"{OUT.stat().st_size:,} bytes", f"{len(OUT.read_text().splitlines()):,} lines")


if __name__ == "__main__":
    sys.exit(main())
