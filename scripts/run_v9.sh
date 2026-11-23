#!/usr/bin/env bash
# v9 fine-tuning (22 Nov). Same size and speed (EfficientNet-B0,
# 224 px, one pass): training-only changes to the v8 recipe. Chosen on PlantDoc out-of-fold + the NEU dev half;
# the NEU test half is scored once at the end. PRE-REGISTERED rule: PS1 Fine-tune v9 Plan,
# applied by scripts/v9_summary.py. v7 stays the submitted model; sealed splits untouched.
#   stage 1: seed 0 of each variant (one change each) vs v8 seed 0 -> winner or stop
#   stage 2: seeds 1-2 of the winner, v9_final, export, decision (t = 0.80), India check, NEU-test, rule
# Run detached:  setsid nohup scripts/run_v9.sh > logs/v9_<date>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
COMMON=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch effnet_b0 --epochs 10 --samples-per-epoch 5000
        --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2
        --exclude-sources maharashtra --ema-decay 0.998 --ema-start-epoch 4
        --extra-manifest data/manifest_extra.csv --extra-oversample 1)          # = v8; a variant's flags come after
declare -A CHANGE=([v9a]="--epochs 20 --ema-start-epoch 8" [v9b]="--backbone-lr-mult 0.3"
                   [v9c]="--backbone-lr-mult 1.0" [v9d]="--extra-oversample 2")
ORDER=(v9b v9c v9d v9a)                    # shortest first; v9a trains twice as long
THRESHOLD=0.8
V8_PTE=runs/v8_final/export/tomato_effnet_b0_t2_fp32_xnnpack.pte

echo 800 > /proc/$$/oom_score_adj
renice -n 10 -p $$ > /dev/null
grep -q ',x_dev_known,' data/manifest_extra.csv || { echo "missing the NEU dev split in data/manifest_extra.csv"; exit 1; }
echo "== start $(date -Is); extra manifest sha $(sha256sum data/manifest_extra.csv | cut -c1-12)"

clean() { grep -v InternalConsistency; }

run_or_stop() { # <log> <command...>: run, show the last lines, stop the whole script if it failed
  local log=$1; shift
  "$@" > "$log" 2>&1 || { clean < "$log" | tail -5; echo "step failed: $*; stopping"; exit 1; }
  clean < "$log" | tail -2
}

neu_dev() {    # <prefix> <seed>
  mkdir -p "runs/$1s$2_ema"
  run_or_stop "runs/$1s$2_ema/neu_dev.log" python -W ignore -m ps1.vision.neu_dev_eval \
    --fold-ckpts "runs/$1s$2_f0/ema.pt" "runs/$1s$2_f1/ema.pt" \
    --abstention "runs/$1s$2_ema/abstention_oe_dev.json" --out "runs/$1s$2_ema"
}

train_seed() { # <prefix> <seed> [variant flags...]
  local p=$1 seed=$2 status=0 pids=() out f q
  shift 2
  for f in 0 1; do
    out="runs/${p}s${seed}_f$f"; mkdir -p "$out"
    python -W ignore -m ps1.vision.train "${COMMON[@]}" "$@" --threads 8 --seed "$seed" --field-fold "$f" --out "$out" \
      > "$out/train.log" 2>&1 &
    pids+=($!)
  done
  for q in "${pids[@]}"; do wait "$q" || status=1; done
  for f in 0 1; do
    out="runs/${p}s${seed}_f$f"
    echo "$out: $(grep -c '"epoch"' "$out/train.log") log lines; ema.pt $([[ -f $out/ema.pt ]] && echo yes || echo MISSING)"
  done
  [[ $status -ne 0 ]] && { echo "$p seed $seed: a training run failed; stopping"; exit 1; }
  mkdir -p "runs/${p}s${seed}_ema"
  run_or_stop "runs/${p}s${seed}_ema/abstain.log" python -W ignore -m ps1.vision.abstain_oe \
    --fold-ckpts "runs/${p}s${seed}_f0/ema.pt" "runs/${p}s${seed}_f1/ema.pt" --out "runs/${p}s${seed}_ema"
  neu_dev "$p" "$seed"
  echo "== $p seed $seed done $(date -Is); free memory: $(free -g | awk '/Mem:/{print $7}') GB"
}

echo "== baselines, before any v9 result $(date -Is)"
run_or_stop runs/v8_final/extra_eval_test.log python -W ignore -m ps1.vision.extra_eval --pte "$V8_PTE" \
  --decision runs/v8_final/export/decision.json --out runs/v8_final --part test
for s in 0 1 2; do neu_dev v8 "$s"; done

for v in "${ORDER[@]}"; do
  echo "== stage 1: $v (${CHANGE[$v]}) seed 0 $(date -Is)"
  # shellcheck disable=SC2086  # the variant's flags are meant to split into words
  train_seed "$v" 0 ${CHANGE[$v]}
done

echo "== stage 1 rule $(date -Is)"
python scripts/v9_summary.py screen > runs/v9_screen.txt || { echo "stage 1 rule failed"; exit 1; }
cat runs/v9_screen.txt
W=$(awk '/^WINNER/{print $2}' runs/v9_screen.txt)
if [[ $W == none || -z $W ]]; then echo "== done (no stage-1 winner: v8 stays) $(date -Is)"; exit 0; fi

for s in 1 2; do
  echo "== stage 2: $W seed $s $(date -Is)"
  # shellcheck disable=SC2086
  train_seed "$W" "$s" ${CHANGE[$W]}
done

OUT=runs/v9_final; EXPORT=$OUT/export; mkdir -p "$OUT"
echo "== final training $W (all non-sealed field data + extra, seed 0) $(date -Is)"
# shellcheck disable=SC2086
python -W ignore -m ps1.vision.train "${COMMON[@]}" ${CHANGE[$W]} --threads 12 --seed 0 --field-fold -1 --out "$OUT" \
  > "$OUT/train.log" 2>&1 || { echo "final training failed, see $OUT/train.log"; exit 1; }
CK=$OUT/ema.pt; [[ -f $CK ]] || { echo "missing $CK"; exit 1; }
echo "== export FP32 from $CK $(date -Is)"
run_or_stop "$OUT/export.log" python -W ignore -m ps1.vision.export --ckpt "$CK" --out "$EXPORT" --precision fp32
PTE=$(ls "$EXPORT"/*_fp32_xnnpack.pte)
python -m ps1.decision --abstention "runs/${W}s0_ema/abstention_oe_dev.json" --sidecar "${PTE%.pte}.json" \
  --threshold "$THRESHOLD" --out "$EXPORT/decision.json" > /dev/null || exit 1
echo "== Indian-photo check $(date -Is)"
run_or_stop "$OUT/india_check.log" python -W ignore -m ps1.vision.india_check --ckpt "$CK" --pte "$PTE" \
  --decision "$EXPORT/decision.json" --out "$OUT"
echo "== v9_final on the NEU test half (first and only look) $(date -Is)"
run_or_stop "$OUT/extra_eval_test.log" python -W ignore -m ps1.vision.extra_eval --pte "$PTE" \
  --decision "$EXPORT/decision.json" --out "$OUT" --part test
echo "== decision rule $(date -Is)"
python scripts/v9_summary.py confirm || { echo "confirm rule failed"; exit 1; }
echo "== done $(date -Is)"
