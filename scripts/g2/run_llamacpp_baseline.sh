#!/usr/bin/env bash
# G2 Experiment D cross-check: same prompt and thread count as the G1 ExecuTorch runs, on llama.cpp, offline.
# Speed + memory only; the weights differ, so never read this as a quality comparison.
# Usage: scripts/g2/run_llamacpp_baseline.sh <file.gguf> [runs=10] [threads=4] [g1_runs.csv to compare with]
# Build first: podman build -t ps1-llamacpp:cd959ba --build-arg JOBS=3 -f scripts/g2/Containerfile.llamacpp scripts/g2
set -euo pipefail

GGUF="${1:?usage: run_llamacpp_baseline.sh <file.gguf> [runs] [threads] [g1_runs.csv]}"
RUNS="${2:-10}"
THREADS="${3:-4}"
G1_CSV="${4:-}"
IMAGE="ps1-llamacpp:cd959ba"
MAX_NEW_TOKENS=120           # same bound as ps1/llm/explain.py
CTX=512                      # same as --seq_len in scripts/g1/run_g1.sh

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ -s "$GGUF" ]] || { echo "missing $GGUF" >&2; exit 1; }
GGUF_DIR="$(cd "$(dirname "$GGUF")" && pwd)"
OUT="$ROOT/logs/llamacpp_$(date +%Y%m%d_%H%M%S)_t${THREADS}"
mkdir -p "$OUT"
sha256sum "$GGUF" > "$OUT/gguf.sha256"

# Keep the prompt byte-identical to G1: $(cat) would strip its trailing newlines.
PROMPT="$(cat "$ROOT/scripts/g1/prompt.txt"; printf x)"
PROMPT="${PROMPT%x}"

for i in $(seq 1 "$RUNS"); do
  echo "run $i/$RUNS"
  podman run --rm --network=none -v "$GGUF_DIR:/models:ro,Z" "$IMAGE" \
    /usr/bin/time -v /opt/llama.cpp/build/bin/llama-completion \
      -m "/models/$(basename "$GGUF")" -p "$PROMPT" -n "$MAX_NEW_TOKENS" -c "$CTX" \
      --temp 0 --seed 1 -t "$THREADS" --no-display-prompt \
    > "$OUT/run_$i.log" 2>&1 || echo "run $i exited non-zero (see $OUT/run_$i.log)"
done

cd "$ROOT"
"${PYTHON:-python3}" -m ps1.llamacpp_baseline "$OUT" ${G1_CSV:+--executorch "$G1_CSV"}
