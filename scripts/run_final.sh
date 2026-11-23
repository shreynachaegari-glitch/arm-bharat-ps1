#!/usr/bin/env bash
# Final candidate model (v6): v5b recipe (no Maharashtra in training) on ALL non-sealed field data, then
# FP32 export, decision config and the Indian-photo check. Run detached (survives closing the terminal):
#   setsid nohup scripts/run_final.sh > logs/final_<date>.log 2>&1 < /dev/null &
# No held-out field data is left, so: temperature = v5b cross-fitted estimate; threshold 0.7 = placeholder
# until Sreyesh confirms the gates; field quality numbers stay the v5b fold estimates. Sealed splits untouched.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
OUT=runs/v6_final
EXPORT=$OUT/export
THRESHOLD=0.7
# Same recipe as v5b; one run only, so 12 threads instead of 8 (memory-safe: 1 trainer + 2 loader workers).
ARGS=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch effnet_b0 --epochs 10 --samples-per-epoch 5000
      --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2 --threads 12 --seed 0
      --field-fold -1 --exclude-sources maharashtra)

echo 800 > /proc/$$/oom_score_adj          # if memory runs short, the kernel kills this before VS Code
renice -n 10 -p $$ > /dev/null

echo "== start $(date -Is)"
mkdir -p "$OUT"
python -W ignore -m ps1.vision.train "${ARGS[@]}" --out "$OUT" > "$OUT/train.log" 2>&1 \
  || { echo "training failed, see $OUT/train.log"; exit 1; }
echo "== trained $(date -Is): $(grep -c '^{' "$OUT/train.log") epochs; last.pt $([[ -f $OUT/last.pt ]] && echo yes || echo MISSING)"
[[ -f $OUT/last.pt ]] || exit 1

echo "== export FP32 $(date -Is)"
python -W ignore -m ps1.vision.export --ckpt "$OUT/last.pt" --out "$EXPORT" --precision fp32 2>&1 \
  | grep -v InternalConsistency | tail -20 || exit 1
PTE=$(ls "$EXPORT"/*_fp32_xnnpack.pte)
echo "== decision config"
python -m ps1.decision --abstention runs/v5b/abstention_oe_dev.json --sidecar "${PTE%.pte}.json" \
  --threshold "$THRESHOLD" --out "$EXPORT/decision.json" || exit 1
echo "== Indian-photo check $(date -Is)"
python -W ignore -m ps1.vision.india_check --ckpt "$OUT/last.pt" --pte "$PTE" --decision "$EXPORT/decision.json" \
  --out "$OUT" 2>&1 | grep -v InternalConsistency
echo "== done $(date -Is)"
