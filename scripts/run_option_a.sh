#!/usr/bin/env bash
# Option A (21 Nov): v5b recipe x seeds 0,1,2 x folds 0,1, each run saving last.pt (raw) and
# ema.pt (EMA decay 0.998 from epoch 4). Then both arms are scored out-of-fold per seed, the PRE-REGISTERED rule
# (PS1 Option A Seeds and EMA Plan) picks the arm, and v7_final is trained, exported,
# bound to a decision config and India-checked. v6_final stays as the fallback. Sealed splits untouched.
# Run detached:  setsid nohup scripts/run_option_a.sh > logs/option_a_<date>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
# v5b recipe + EMA; memory-safe (2 runs at a time, 2 loader workers, 8 threads each).
COMMON=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch effnet_b0 --epochs 10 --samples-per-epoch 5000
        --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2
        --exclude-sources maharashtra --ema-decay 0.998 --ema-start-epoch 4)
THRESHOLD=0.7          # placeholder until Sreyesh confirms the gates

echo 800 > /proc/$$/oom_score_adj          # if memory runs short, the kernel kills this before VS Code
renice -n 10 -p $$ > /dev/null

echo "== start $(date -Is)"
for seed in 0 1 2; do
  pids=()
  for f in 0 1; do
    out="runs/v7s${seed}_f$f"; mkdir -p "$out"
    python -W ignore -m ps1.vision.train "${COMMON[@]}" --threads 8 --seed "$seed" --field-fold "$f" --out "$out" \
      > "$out/train.log" 2>&1 &
    pids+=($!)
  done
  status=0
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  for f in 0 1; do
    d="runs/v7s${seed}_f$f"
    echo "$d: $(grep -c '"epoch"' "$d/train.log") log lines; last.pt $([[ -f $d/last.pt ]] && echo yes || echo MISSING); ema.pt $([[ -f $d/ema.pt ]] && echo yes || echo MISSING)"
  done
  [[ $status -ne 0 ]] && { echo "seed $seed: a training run failed; stopping"; exit 1; }
  for arm in raw ema; do
    ck=last.pt; [[ $arm == ema ]] && ck=ema.pt
    echo "== abstention seed $seed $arm $(date -Is)"
    python -W ignore -m ps1.vision.abstain_oe --fold-ckpts "runs/v7s${seed}_f0/$ck" "runs/v7s${seed}_f1/$ck" \
        --out "runs/v7s${seed}_$arm" 2>&1 | grep -v InternalConsistency | head -3
  done
  echo "== seed $seed done $(date -Is); free memory: $(free -g | awk '/Mem:/{print $7}') GB"
done

echo "== decision rule $(date -Is)"
ARM=$(python scripts/option_a_summary.py --print-arm) || { echo "summary failed"; exit 1; }
python scripts/option_a_summary.py
echo "== chosen arm: $ARM"

OUT=runs/v7_final; EXPORT=$OUT/export; mkdir -p "$OUT"
echo "== final training (all non-sealed field data, seed 0) $(date -Is)"
python -W ignore -m ps1.vision.train "${COMMON[@]}" --threads 12 --seed 0 --field-fold -1 --out "$OUT" > "$OUT/train.log" 2>&1 \
  || { echo "final training failed, see $OUT/train.log"; exit 1; }
CK=$OUT/last.pt; [[ $ARM == ema ]] && CK=$OUT/ema.pt
[[ -f $CK ]] || { echo "missing $CK"; exit 1; }
echo "== export FP32 from $CK $(date -Is)"
python -W ignore -m ps1.vision.export --ckpt "$CK" --out "$EXPORT" --precision fp32 2>&1 | grep -v InternalConsistency | tail -6 || exit 1
PTE=$(ls "$EXPORT"/*_fp32_xnnpack.pte)
echo "== decision config (temperature from runs/v7s0_$ARM)"
python -m ps1.decision --abstention "runs/v7s0_$ARM/abstention_oe_dev.json" --sidecar "${PTE%.pte}.json" \
  --threshold "$THRESHOLD" --out "$EXPORT/decision.json" || exit 1
echo "== Indian-photo check $(date -Is)"
python -W ignore -m ps1.vision.india_check --ckpt "$CK" --pte "$PTE" --decision "$EXPORT/decision.json" --out "$OUT" 2>&1 \
  | grep -v InternalConsistency
echo "== done $(date -Is)"
