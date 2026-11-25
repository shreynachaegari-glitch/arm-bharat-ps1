#!/usr/bin/env bash
# Start the app on the phone, then open http://127.0.0.1:5000 in the phone's browser. Works in aeroplane mode.
cd "$(dirname "$0")"
command -v termux-wake-lock > /dev/null && termux-wake-lock
EXPORT=runs/v7_final/export
exec ./enter.sh python -W ignore -m ps1.app --pte $EXPORT/tomato_effnet_b0_t2_fp32_xnnpack.pte \
  --decision $EXPORT/decision.json --allow-drafts --llama "$@"
