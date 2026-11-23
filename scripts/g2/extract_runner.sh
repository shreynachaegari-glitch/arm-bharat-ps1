#!/usr/bin/env bash
# Take llama_main out of the runner image so a request pays no container start-up. It is not standalone: it links
# libllama_runner.so and libextension_module.so from the build tree, so those come too, and pi-bin/llama_main is a
# small wrapper that points at them and then becomes the binary (exec keeps timeouts and kills working).
# Usage: scripts/g2/extract_runner.sh [out_dir=pi-bin] [image=ps1-et:0.6.0]
set -euo pipefail
OUT="${1:-pi-bin}"
IMAGE="${2:-ps1-et:0.6.0}"
mkdir -p "$OUT/lib"
OUT="$(cd "$OUT" && pwd)"

podman run --rm -v "$OUT:/out:Z" "$IMAGE" sh -c '
  set -eu
  bin=/opt/executorch/cmake-out/examples/models/llama/llama_main
  cp "$bin" /out/llama_main.bin
  ldd "$bin" | awk "\$3 ~ \"^/opt/executorch/\" {print \$3}" | while read -r lib; do cp "$lib" /out/lib/; done
  # Libraries the dynamic loader could not place from the build tree must not pass silently.
  if ldd "$bin" | grep -q "not found"; then ldd "$bin" | grep "not found"; exit 1; fi
'

cat > "$OUT/llama_main" <<'WRAPPER'
#!/bin/sh
HERE="$(cd "$(dirname "$0")" && pwd)"
LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" exec "$HERE/llama_main.bin" "$@"
WRAPPER
chmod +x "$OUT/llama_main" "$OUT/llama_main.bin"

if ! "$OUT/llama_main" --help > /dev/null 2>"$OUT/check.err"; then
  # This image may be for another OS/arch than the caller's shell: say so, do not call it a success.
  echo "WARNING: $OUT/llama_main does not start here:"; cat "$OUT/check.err"; exit 1
fi
rm -f "$OUT/check.err"
echo "ok: $OUT/llama_main ($(ls "$OUT/lib" | wc -l) libraries)"
