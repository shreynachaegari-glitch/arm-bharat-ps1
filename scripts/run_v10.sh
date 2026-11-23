#!/usr/bin/env bash
# v10 bigger backbone (23 Nov; vision up to ~3x B0 on the Pi). v10 = the v8 recipe + --arch effnet_b2 (input 260 px follows the arch). Scored like v9: PlantDoc
# out-of-fold + NEU dev half; NEU test half looked at once at the end. PRE-REGISTERED rule:
# PS1 Bigger Backbone v10 Plan, applied by scripts/v10_summary.py. v7 stays submitted; sealed untouched.
#   stage A: seeds 0-2 (folds 0/1) -> 3-seed means vs v8 -> PASS or stop
#   stage B: v10_final, export (speed + parity), decision (t = 0.80), India check, NEU-test, rule
# Resumable: a fold whose history.json exists (train.py writes it last) is not retrained; the evaluators are re-run.
# Needs mains power (23 Nov: the first launch was on battery and died with the laptop after 2 epochs).
# Run detached:  setsid nohup scripts/run_v10.sh > logs/v10_<date>.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
RECIPE=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch effnet_b2 --epochs 10 --samples-per-epoch 5000
        --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2
        --exclude-sources maharashtra --ema-decay 0.998 --ema-start-epoch 4
        --extra-manifest data/manifest_extra.csv --extra-oversample 1)          # = v8 except --arch
P=v10
THRESHOLD=0.8

on_ac() { grep -qs 1 /sys/class/power_supply/A*/online; }
on_ac || { echo "on battery: plug the laptop in first; stopping"; exit 1; }
STAGGER=150        # s between the two folds, so their end-of-epoch evaluations do not overlap (memory)

echo 800 > /proc/$$/oom_score_adj
renice -n 10 -p $$ > /dev/null
echo "== start $(date -Is); extra manifest sha $(sha256sum data/manifest_extra.csv | cut -c1-12)"

clean() { grep -v InternalConsistency; }

run_or_stop() { # <log> <command...>: run, show the last lines, stop the whole script if it failed
  local log=$1; shift
  "$@" > "$log" 2>&1 || { clean < "$log" | tail -5; echo "step failed: $*; stopping"; exit 1; }
  clean < "$log" | tail -2
}

train_seed() { # <seed>
  local seed=$1 status=0 pids=() out f q
  for f in 1 0; do               # fold 0 is the slower one: starting it second keeps the gap from closing
    out="runs/${P}s${seed}_f$f"; mkdir -p "$out"
    [[ -f $out/history.json ]] && { echo "$out: already trained, kept"; continue; }
    [[ ${#pids[@]} -gt 0 ]] && sleep "$STAGGER"
    python -W ignore -m ps1.vision.train "${RECIPE[@]}" --threads 8 --seed "$seed" --field-fold "$f" --out "$out" \
      > "$out/train.log" 2>&1 &
    pids+=($!)
  done
  for q in "${pids[@]}"; do wait "$q" || status=1; done
  for f in 0 1; do
    out="runs/${P}s${seed}_f$f"
    echo "$out: $(grep -c '"epoch"' "$out/train.log") log lines; ema.pt $([[ -f $out/ema.pt ]] && echo yes || echo MISSING)"
  done
  [[ $status -ne 0 ]] && { echo "$P seed $seed: a training run failed; stopping"; exit 1; }
  mkdir -p "runs/${P}s${seed}_ema"
  run_or_stop "runs/${P}s${seed}_ema/abstain.log" python -W ignore -m ps1.vision.abstain_oe \
    --fold-ckpts "runs/${P}s${seed}_f0/ema.pt" "runs/${P}s${seed}_f1/ema.pt" --out "runs/${P}s${seed}_ema"
  run_or_stop "runs/${P}s${seed}_ema/neu_dev.log" python -W ignore -m ps1.vision.neu_dev_eval \
    --fold-ckpts "runs/${P}s${seed}_f0/ema.pt" "runs/${P}s${seed}_f1/ema.pt" \
    --abstention "runs/${P}s${seed}_ema/abstention_oe_dev.json" --out "runs/${P}s${seed}_ema"
  echo "== $P seed $seed done $(date -Is); free memory: $(free -g | awk '/Mem:/{print $7}') GB"
}

for s in 0 1 2; do
  echo "== stage A: $P seed $s $(date -Is)"
  train_seed "$s"
done

echo "== stage A rule $(date -Is)"
python scripts/v10_summary.py seeds > runs/v10_seeds.txt || { echo "stage A rule failed"; exit 1; }
cat runs/v10_seeds.txt
if [[ $(tail -1 runs/v10_seeds.txt) != PASS ]]; then echo "== done (stage A failed: v8 stays) $(date -Is)"; exit 0; fi

OUT=runs/v10_final; EXPORT=$OUT/export; mkdir -p "$OUT"
echo "== final training (all non-sealed field data + extra, seed 0) $(date -Is)"
[[ -f $OUT/history.json ]] || python -W ignore -m ps1.vision.train "${RECIPE[@]}" --threads 12 --seed 0 --field-fold -1 \
  --out "$OUT" > "$OUT/train.log" 2>&1 || { echo "final training failed, see $OUT/train.log"; exit 1; }
CK=$OUT/ema.pt; [[ -f $CK ]] || { echo "missing $CK"; exit 1; }
echo "== export FP32 from $CK $(date -Is)"
run_or_stop "$OUT/export.log" python -W ignore -m ps1.vision.export --ckpt "$CK" --out "$EXPORT" --precision fp32
PTE=$(ls "$EXPORT"/*_fp32_xnnpack.pte)
python -m ps1.decision --abstention "runs/${P}s0_ema/abstention_oe_dev.json" --sidecar "${PTE%.pte}.json" \
  --threshold "$THRESHOLD" --out "$EXPORT/decision.json" > /dev/null || exit 1
echo "== Indian-photo check $(date -Is)"
run_or_stop "$OUT/india_check.log" python -W ignore -m ps1.vision.india_check --ckpt "$CK" --pte "$PTE" \
  --decision "$EXPORT/decision.json" --out "$OUT"
echo "== v10_final on the NEU test half (first and only look) $(date -Is)"
run_or_stop "$OUT/extra_eval_test.log" python -W ignore -m ps1.vision.extra_eval --pte "$PTE" \
  --decision "$EXPORT/decision.json" --out "$OUT" --part test
echo "== decision rule $(date -Is)"
python scripts/v10_summary.py confirm || { echo "confirm rule failed"; exit 1; }
echo "== done $(date -Is)"
