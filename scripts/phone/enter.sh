#!/usr/bin/env bash
# Run a command inside the bundle's Arm64 Debian. Usage: ./enter.sh python -V
HERE="$(cd "$(dirname "$0")" && pwd)"
command -v proot > /dev/null || { echo "proot is missing: run  pkg install proot" >&2; exit 1; }
unset LD_PRELOAD                       # Termux's own preload library does not exist inside Debian
mkdir -p "$HERE/rootfs/tmp"
KILL=()                                # Termux's proot can stop the app with the window; other builds lack the option
proot --help 2>&1 | grep -q -- --kill-on-exit && KILL=(--kill-on-exit)
# PROOT_EXTRA is only for the emulated test on the dev box.
exec proot -0 "${KILL[@]}" ${PROOT_EXTRA:-} -r "$HERE/rootfs" -b /dev -b /proc -b /sys -b "$HERE/app:/app" -w /app \
  /usr/bin/env -i HOME=/root TMPDIR=/tmp LANG=C.UTF-8 PATH=/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=/opt/site:/app PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=/usr/local/lib:/opt/site/torch/lib \
  PS1_LLAMA_MAIN=/app/pi-bin/llama_main "$@"
