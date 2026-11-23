#!/usr/bin/env bash
# G1 smoke test: run the pinned Llama .pte through the pinned llama_main N times, offline.
# Usage: scripts/g1/run_g1.sh [runs=10] [threads=4] [image=ps1-et:0.6.0]
set -euo pipefail

RUNS="${1:-10}"
THREADS="${2:-4}"          # 4 = Raspberry Pi 5 core count, so x86 numbers are comparable in shape
IMAGE="${3:-ps1-et:0.6.0}"
SEQ_LEN=512

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="$ROOT/artifacts/llama"
OUT="$ROOT/logs/g1_$(date +%Y%m%d_%H%M%S)_t${THREADS}"
mkdir -p "$OUT"
cp "$ROOT/scripts/g1/prompt.txt" "$OUT/prompt.txt"

for f in Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte tokenizer.model; do
  [[ -s "$MODEL_DIR/$f" ]] || { echo "missing $MODEL_DIR/$f" >&2; exit 1; }
done

# $(cat) strips trailing newlines; the production prompt (ps1/llm/explain.py) ends with "\n\n" after
# the assistant header, so append a sentinel and remove it to keep the prompt byte-identical.
PROMPT="$(cat "$OUT/prompt.txt"; printf x)"
PROMPT="${PROMPT%x}"

for i in $(seq 1 "$RUNS"); do
  echo "run $i/$RUNS"
  # --network=none: the runner must work with no network at all (v4 acceptance checklist).
  podman run --rm --network=none \
    -v "$MODEL_DIR:/models:ro,Z" -v "$OUT:/out:Z" "$IMAGE" \
    /usr/bin/time -v llama_main \
      --model_path=/models/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte \
      --tokenizer_path=/models/tokenizer.model \
      --prompt="$PROMPT" \
      --seq_len="$SEQ_LEN" --temperature=0 --cpu_threads="$THREADS" \
    > "$OUT/run_$i.log" 2>&1 || echo "run $i exited non-zero (see $OUT/run_$i.log)"
done

python3 "$ROOT/scripts/g1/parse_g1.py" "$OUT"
