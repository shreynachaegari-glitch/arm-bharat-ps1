#!/usr/bin/env bash
# Start the offline web app with the submitted model (v7_final; v6_final is the fallback). Usage:
#   scripts/run_app.sh                  # this device's screen only: http://127.0.0.1:5000
#   scripts/run_app.sh --host 0.0.0.0   # also phones on the Pi's own Wi-Fi hotspot (no internet needed)
#   scripts/run_app.sh --no-llama       # Screen 1 only (cards), no Llama explanation
#   scripts/run_app.sh --model v8       # v8_final (v7 recipe + Bangladesh field data, after the sealed test;
#                                       # report section 5.4); the page shows a banner saying so
# Cards are still drafts (not expert-reviewed), so --allow-drafts is passed and the page shows a DRAFT banner.
# On the Pi, Llama uses the native llama_main (PS1_LLAMA_MAIN or PATH); on x86 it uses the G1 container.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL=v7
LLAMA=(--llama)
ARGS=()
while (($#)); do
  case $1 in
    --no-llama) LLAMA=() ;;
    --model) MODEL=${2:-}; shift ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
NOTE=()
case $MODEL in
  v7) ;;
  v8) NOTE=(--model-note "Model v8: trained with extra Bangladesh field photos after the sealed test. The submitted Phase 1 model is v7.") ;;
  *) echo "unknown --model '$MODEL' (use v7 or v8)" >&2; exit 1 ;;
esac
EXPORT=runs/${MODEL}_final/export
PTE=$EXPORT/tomato_effnet_b0_t2_fp32_xnnpack.pte
[[ -f $PTE && -f $EXPORT/decision.json ]] || { echo "missing $PTE or its decision.json" >&2; exit 1; }
exec python -W ignore -m ps1.app --pte "$PTE" --decision "$EXPORT/decision.json" --allow-drafts "${LLAMA[@]}" "${NOTE[@]}" "${ARGS[@]}"
