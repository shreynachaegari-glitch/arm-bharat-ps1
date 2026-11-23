#!/usr/bin/env bash
# Maharashtra-shortcut ablation + follow-ups, meant to run detached (survives closing the terminal):
#   setsid nohup scripts/run_ablation.sh > logs/ablation_<date>.log 2>&1 < /dev/null &
# v5a = v4 recipe on current code; v5b = same minus the Maharashtra training images.
# Afterwards (idle CPU): ablation verdict, G1 10-run Llama benchmark, x86 latency matrix.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

C=tomato_early_blight,tomato_healthy,tomato_late_blight,tomato_leaf_spot,tomato_other_problem
# Memory-safe settings (31 Oct: 4 parallel runs x 10 processes used 28/31 GB and the OOM killer took VS Code):
# 2 runs at a time, 2 loader workers, 8 threads each.
COMMON=(--classes "$C" --taxonomy t2 --fold-scheme stratified --arch effnet_b0 --epochs 10 --samples-per-epoch 5000
        --field-oversample 8 --composite-p 0.5 --backbone-lr-mult 0.1 --batch 32 --workers 2 --threads 8 --seed 0)

# Children inherit these: if memory runs short, the kernel kills training before VS Code (oom_score_adj 300).
echo 800 > /proc/$$/oom_score_adj
renice -n 10 -p $$ > /dev/null

echo "== start $(date -Is)"
status=0
for arm in a b; do                       # one arm (both folds) at a time
  pids=()
  for f in 0 1; do
    out="runs/v5${arm}_f$f"; mkdir -p "$out"
    extra=(); [[ $arm == b ]] && extra=(--exclude-sources maharashtra)
    python -W ignore -m ps1.vision.train "${COMMON[@]}" --field-fold "$f" "${extra[@]}" --out "$out" > "$out/train.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || status=1; done
  echo "== arm $arm finished $(date -Is) (status $status); free memory: $(free -g | awk '/Mem:/{print $7}') GB"
done
echo "== training finished $(date -Is) (status $status)"
for d in runs/v5a_f0 runs/v5a_f1 runs/v5b_f0 runs/v5b_f1; do
  echo "$d: $(grep -c '^{' "$d/train.log") epochs; last.pt $([[ -f $d/last.pt ]] && echo yes || echo MISSING)"
done
[[ $status -ne 0 ]] && { echo "a training run failed; stopping before evaluation"; exit 1; }

for arm in a b; do
  echo "== abstention v5$arm"
  python -W ignore -m ps1.vision.abstain_oe --fold-ckpts "runs/v5${arm}_f0/last.pt" "runs/v5${arm}_f1/last.pt" \
      --out "runs/v5$arm" 2>&1 | grep -v InternalConsistency | tail -11
done
echo "== G1 10-run $(date -Is)"
scripts/g1/run_g1.sh 10 4 2>&1 | tail -12
echo "== latency matrix $(date -Is)"
python -W ignore -m ps1.bench --matrix scripts/g2/bench_matrix.json --only vision_fp32,full_fp32_llama_t4,full_fp32_llama_t2 \
    --images runs/bench/images --out runs/bench/x86_idle 2>&1 | grep -v InternalConsistency
echo "== done $(date -Is)"
