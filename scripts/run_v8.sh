#!/usr/bin/env bash
# v8 (21 Nov): v7 recipe (v5b + EMA) + extra field data
# from data/manifest_extra.csv (DIU Bangladesh, training only). Seeds 0-2 x folds 0-1 scored out-of-fold on the same
# PlantDoc dev folds as v7; then v8_final, export, decision (T from v8s0_ema, threshold 0.80 not re-tuned), India check,
# and the held-out NEU test for v8_final AND v7_final. The PRE-REGISTERED rule (PS1 Extra Field Data
# v8 Plan) is applied by scripts/v8_summary.py. v7 stays the submitted model. Sealed splits untouched.
# Run detached:  setsid nohup scripts/run_v8.sh > logs/v8_<date>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
COMMON=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch effnet_b0 --epochs 10 --samples-per-epoch 5000
        --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2
        --exclude-sources maharashtra --ema-decay 0.998 --ema-start-epoch 4
        --extra-manifest data/manifest_extra.csv --extra-oversample 1)
THRESHOLD=0.8

echo 800 > /proc/$$/oom_score_adj
renice -n 10 -p $$ > /dev/null
[[ -f data/manifest_extra.csv ]] || { echo "missing data/manifest_extra.csv"; exit 1; }
echo "== start $(date -Is); extra manifest sha $(sha256sum data/manifest_extra.csv | cut -c1-12)"

echo "== v7_final on the held-out NEU test (before any v8 result) $(date -Is)"
python -W ignore -m ps1.vision.extra_eval --pte runs/v7_final/export/tomato_effnet_b0_t2_fp32_xnnpack.pte \
  --decision runs/v7_final/export/decision.json --out runs/v7_final --part all 2>&1 | grep -v InternalConsistency | tail -1

for seed in 0 1 2; do
  pids=()
  for f in 0 1; do
    out="runs/v8s${seed}_f$f"; mkdir -p "$out"
    python -W ignore -m ps1.vision.train "${COMMON[@]}" --threads 8 --seed "$seed" --field-fold "$f" --out "$out" \
      > "$out/train.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  for f in 0 1; do
    d="runs/v8s${seed}_f$f"
    echo "$d: $(grep -c '"epoch"' "$d/train.log") log lines; ema.pt $([[ -f $d/ema.pt ]] && echo yes || echo MISSING)"
  done
  [[ $status -ne 0 ]] && { echo "seed $seed: a training run failed; stopping"; exit 1; }
  echo "== abstention seed $seed ema $(date -Is)"
  python -W ignore -m ps1.vision.abstain_oe --fold-ckpts "runs/v8s${seed}_f0/ema.pt" "runs/v8s${seed}_f1/ema.pt" \
      --out "runs/v8s${seed}_ema" 2>&1 | grep -v InternalConsistency | head -3
  echo "== seed $seed done $(date -Is); free memory: $(free -g | awk '/Mem:/{print $7}') GB"
done

OUT=runs/v8_final; EXPORT=$OUT/export; mkdir -p "$OUT"
echo "== final training (all non-sealed field data + extra, seed 0) $(date -Is)"
python -W ignore -m ps1.vision.train "${COMMON[@]}" --threads 12 --seed 0 --field-fold -1 --out "$OUT" > "$OUT/train.log" 2>&1 \
  || { echo "final training failed, see $OUT/train.log"; exit 1; }
CK=$OUT/ema.pt; [[ -f $CK ]] || { echo "missing $CK"; exit 1; }
echo "== export FP32 from $CK $(date -Is)"
python -W ignore -m ps1.vision.export --ckpt "$CK" --out "$EXPORT" --precision fp32 2>&1 | grep -v InternalConsistency | tail -6 || exit 1
PTE=$(ls "$EXPORT"/*_fp32_xnnpack.pte)
python -m ps1.decision --abstention runs/v8s0_ema/abstention_oe_dev.json --sidecar "${PTE%.pte}.json" \
  --threshold "$THRESHOLD" --out "$EXPORT/decision.json" > /dev/null || exit 1
echo "== Indian-photo check $(date -Is)"
python -W ignore -m ps1.vision.india_check --ckpt "$CK" --pte "$PTE" --decision "$EXPORT/decision.json" --out "$OUT" 2>&1 \
  | grep -v InternalConsistency | tail -3
echo "== v8_final on the held-out NEU test $(date -Is)"
python -W ignore -m ps1.vision.extra_eval --pte "$PTE" --decision "$EXPORT/decision.json" --out "$OUT" --part all 2>&1 \
  | grep -v InternalConsistency | tail -1
echo "== decision rule $(date -Is)"
python scripts/v8_summary.py
echo "== done $(date -Is)"
