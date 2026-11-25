# Grounded Field Assistant (Arm Bharat AI-SoC 2026-27, PS1)

An offline assistant that checks a photo of a tomato leaf and gives safe, sourced advice. It is built for a Raspberry Pi 5 (Arm CPU) and needs no internet:

- **Vision:** an EfficientNet-B0 model exported to ExecuTorch (XNNPACK) names the likely problem, or says it is not sure.
- **Advice:** a reviewed advice card gives the actions.
- **Explanation:** **Llama 3.2 1B (INT4) running on ExecuTorch** rewrites the card's text in simpler words. The output is checked before it is shown.

Full write-up: **[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)** · Demo video: [docs/demo/ps1_app_demo_x86_20261121.mp4](docs/demo/ps1_app_demo_x86_20261121.mp4) · Screens: [docs/app-screens/](docs/app-screens/)

> Phase 1 status: everything runs end to end on an x86 Linux laptop. The Raspberry Pi 5 run (G2) is pending hardware; its build kit and runbook are in [scripts/g2/README.md](scripts/g2/README.md).

## What it does

```
photo → quality gate → vision .pte → decision (or "no clear match") → advice card → Screen 1
                                                       └→ Llama .pte rewrites the card text → checker → Screen 2
```

- **Classes:** early blight, late blight, leaf spot (bacterial spot or Septoria), healthy, and "other problem".
- **Safety:** no pesticide names or doses. The app says when a photo cannot confirm a problem and when to see an extension officer.
- **Privacy:** the photo is deleted after use, and the page loads nothing from the internet.

## Quick start (x86 Linux, Python 3.12)

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
                                             # executorch 0.6.0, torch 2.7.0 (CPU), flask, waitress, ...
python -m pytest -q                          # 273 tests
```

**Llama model** (not in git: 1.1 GB). Download the pinned revision into `artifacts/llama/`:

```bash
hf download executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET \
  --revision 911221e0 --local-dir artifacts/llama
sha256sum artifacts/llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte   # expect 8715cdba9e91f6be...cda3b
```

On x86 the `llama_main` runner is built in a container (`scripts/g1/`). On the Pi it is built natively (`scripts/g2/README.md`).

**Run the app:**

```bash
scripts/run_app.sh                  # http://127.0.0.1:5000 on this device
scripts/run_app.sh --host 0.0.0.0   # also phones on the Pi's own Wi-Fi hotspot
scripts/run_app.sh --no-llama       # cards only (no Llama)
scripts/run_app.sh --model v8       # v8 (extra Bangladesh field data, after the sealed test); page says so
```

The advice cards are drafts until an agriculture expert reviews them, and the page shows a "Draft advice" banner until then.

## Repository layout

| Path | What |
|---|---|
| `ps1/contracts.py` | typed objects passed between stages; the safety rules are enforced in their validators |
| `ps1/quality_gate.py`, `ps1/decision.py` | photo quality checks; temperature + threshold + abstention, bound to the model hash |
| `ps1/vision/` | training, export to `.pte`, runtime, abstention, sealed final evaluation |
| `ps1/llm/` | Llama wrapper (buffered, timeout, rewrite prompt) and the checker for generated text |
| `ps1/cards.py`, `cards/en/` | advice cards (sources, actions, possible causes) |
| `ps1/pipeline.py` | the end-to-end pipeline with per-stage timers |
| `ps1/app/` | local web app (Flask + waitress, offline page) |
| `scripts/` | training/export runs, Pi kit (`g2/`), Llama card check, report generator |
| `runs/` | result artifacts (JSON/CSV); the submitted model `runs/v7_final/export/` (v6_final kept as fallback); `runs/v8_final/export/` optional (report §5.4) |
| `tests/` | 273 tests |

## Data

- **PlantVillage** (CC BY-SA 3.0): lab-style training.
- **PlantDoc** (CC BY 4.0): field photos, half of which is a sealed test.
- **Maharashtra tomato dataset** (Mendeley `rv3kxfv47y`, CC BY 4.0): Indian photos, kept out of training as a check.

The datasets are not stored in git. `ps1/data/build_splits.py` rebuilds the pinned manifest.

## Licences

This repository has no separate code licence file. Llama 3.2 is used under the Llama 3.2 Community License ("Built with Llama"). Dataset licences are listed above.
