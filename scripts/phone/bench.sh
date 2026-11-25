#!/usr/bin/env bash
# Checks and measurements on the phone. Close other apps, plug in, keep the screen on. Results go to app/results/.
# Usage: ./bench.sh
cd "$(dirname "$0")"
OUT=app/results
mkdir -p $OUT
command -v termux-wake-lock > /dev/null && termux-wake-lock
{ date; command -v getprop > /dev/null && { getprop ro.product.manufacturer; getprop ro.product.model; getprop ro.build.version.release; getprop ro.soc.model; }
  uname -m; grep MemTotal /proc/meminfo; } | tee $OUT/device.txt
MODEL=runs/v7_final/export/tomato_effnet_b0_t2_fp32_xnnpack.pte
set -x
./enter.sh python -W ignore -m ps1.vision.golden --check --pte $MODEL --golden runs/v7_final/golden.json \
  --report results/golden.json 2>&1 | tee $OUT/golden.log
./enter.sh python -W ignore -m ps1.bench --matrix scripts/g2/bench_matrix.json --images runs/bench/images \
  --only vision_fp32,full_fp32_llama_t4,full_fp32_llama_t2 --out results/bench 2>&1 | tee $OUT/bench.log
./enter.sh python scripts/phone/llama_runs.py 10 4 results/llama_t4 2>&1 | tee $OUT/llama_t4.log
tar -czf ps1-phone-results.tar.gz -C app results
set +x
echo "Done. Copy it to the phone's Download folder with:  cp ps1-phone-results.tar.gz ~/storage/downloads/"
