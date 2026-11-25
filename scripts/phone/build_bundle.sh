#!/usr/bin/env bash
# Build the Android bundle on the x86 dev box: one .tar.gz with an Arm64 Debian + Python 3.12, ExecuTorch, both models
# and the app. Nothing is compiled on the phone. Usage: scripts/phone/build_bundle.sh   (about 10 minutes, ~3 GB free)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=build/phone
KIT="$OUT/ps1-phone"
[[ ! -e $OUT ]] || { echo "$OUT already exists: move or remove it first" >&2; exit 1; }
for f in artifacts/llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte artifacts/llama/tokenizer.model runs/v7_final/golden.json; do
  [[ -s $f ]] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p "$OUT/cross" "$KIT/rootfs" "$KIT/app"

podman build -q -t ps1-cross:0.6.0 -f scripts/phone/Containerfile.cross scripts/phone
podman run --rm -v "$PWD/scripts/phone:/kit:ro,z" -v "$PWD/$OUT/cross:/out:Z" ps1-cross:0.6.0 bash /kit/cross_build.sh

echo "== root file system"
podman pull -q --platform linux/arm64 docker.io/library/python:3.12-slim-bookworm
container="$(podman create --platform linux/arm64 docker.io/library/python:3.12-slim-bookworm)"
podman export "$container" | tar -x -C "$KIT/rootfs" --no-same-owner
podman rm "$container" > /dev/null
mkdir -p "$KIT/rootfs/opt" "$KIT/rootfs/app"
mv "$OUT/cross/site" "$KIT/rootfs/opt/site"

echo "== app"
tar -c --exclude='__pycache__' --exclude='scripts/update_refs.sh' ps1 cards tests scripts pyproject.toml | tar -x -C "$KIT/app"
mkdir -p "$KIT/app/runs/v7_final" "$KIT/app/runs/bench/images" "$KIT/app/artifacts/llama"
cp -r runs/v7_final/export runs/v7_final/golden.json "$KIT/app/runs/v7_final/"
cp -L runs/bench/images/*.jpg "$KIT/app/runs/bench/images/"
cp artifacts/llama/*.pte artifacts/llama/tokenizer.model "$KIT/app/artifacts/llama/"
cp -r "$OUT/cross/pi-bin" "$KIT/app/pi-bin"
cp scripts/phone/{enter.sh,start.sh,bench.sh,README.md} "$KIT/"

echo "== pack"
# Android forbids hard links for apps, so duplicates are stored as plain copies.
tar -C "$OUT" --hard-dereference --owner=0 --group=0 --mode='u+rwX' -czf "$OUT/ps1-phone.tar.gz" ps1-phone
(cd "$OUT" && sha256sum ps1-phone.tar.gz | tee ps1-phone.tar.gz.sha256 && ls -lh ps1-phone.tar.gz)
