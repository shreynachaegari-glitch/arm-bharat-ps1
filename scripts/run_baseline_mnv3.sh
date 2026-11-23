#!/usr/bin/env bash
# Baseline comparison (21 Nov): MobileNetV3-Small with the exact v5b recipe, so the report can show
# MobileNetV3 vs EfficientNet-B0 on the same dev folds. Only --arch differs from v5b. Sealed splits untouched.
# Run detached (survives closing the terminal):
#   setsid nohup scripts/run_baseline_mnv3.sh > logs/baseline_mnv3_<date>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
# v5b COMMON with --arch mnv3s; memory-safe (2 runs, 2 loader workers, 8 threads each).
COMMON=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch mnv3s --epochs 10 --samples-per-epoch 5000
        --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2 --threads 8 --seed 0
        --exclude-sources maharashtra)

echo 800 > /proc/$$/oom_score_adj          # if memory runs short, the kernel kills this before VS Code
renice -n 10 -p $$ > /dev/null

echo "== start $(date -Is)"
status=0
pids=()
for f in 0 1; do
  out="runs/v5m_f$f"; mkdir -p "$out"
  python -W ignore -m ps1.vision.train "${COMMON[@]}" --field-fold "$f" --out "$out" > "$out/train.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || status=1; done
for d in runs/v5m_f0 runs/v5m_f1; do
  echo "$d: $(grep -c '^{' "$d/train.log") epochs; last.pt $([[ -f $d/last.pt ]] && echo yes || echo MISSING)"
done
[[ $status -ne 0 ]] && { echo "a training run failed; stopping before evaluation"; exit 1; }

echo "== abstention v5m (compare with runs/v5b/abstention_oe_dev.json)"
python -W ignore -m ps1.vision.abstain_oe --fold-ckpts runs/v5m_f0/last.pt runs/v5m_f1/last.pt \
    --out runs/v5m 2>&1 | grep -v InternalConsistency | tail -11
echo "== FP32 export of fold 0 (size and latency only) $(date -Is)"
python -W ignore -m ps1.vision.export --ckpt runs/v5m_f0/last.pt --out runs/v5m_f0/export --precision fp32 2>&1 \
  | grep -v InternalConsistency | tail -8
echo "== done $(date -Is)"
