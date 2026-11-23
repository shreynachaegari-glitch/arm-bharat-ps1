#!/usr/bin/env bash
# G2 preflight: is this Pi (and its files) fit to give benchmark numbers we can report? Read-only; changes nothing.
# Run from anywhere: scripts/g2/preflight.sh     Exit 0 = no FAIL lines (WARN lines still go in the log).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fails=0
ok()   { echo "ok    $1"; }
warn() { echo "WARN  $1"; }
fail() { echo "FAIL  $1"; fails=$((fails + 1)); }

MIN_RAM_KB=7000000       # the 8 GB board reports a little under 8 GB
MIN_DISK_GB=10           # Llama .pte 1.1 GB, container images ~5 GB, room for logs
HOT_C=70                 # above this at idle, cooling is not adequate for sustained numbers

[[ "$(uname -m)" == "aarch64" ]] && ok "arch aarch64" || fail "arch $(uname -m), expected aarch64 (a 64-bit Pi OS)"
model=unknown
[[ -r /proc/device-tree/model ]] && model="$(tr -d '\0' < /proc/device-tree/model)"
[[ "$model" == *"Raspberry Pi 5"* ]] && ok "board: $model" || fail "board: $model, expected Raspberry Pi 5"
ok "os: $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}"); kernel $(uname -r)"
[[ "$(nproc)" == "4" ]] && ok "cores 4" || warn "cores $(nproc), Pi 5 has 4"

ram_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
(( ram_kb >= MIN_RAM_KB )) && ok "ram $((ram_kb / 1024)) MB" || fail "ram $((ram_kb / 1024)) MB, expected the 8 GB board"
disk_gb="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')"
(( disk_gb >= MIN_DISK_GB )) && ok "disk free ${disk_gb} GB" || fail "disk free ${disk_gb} GB, need ${MIN_DISK_GB}"

load="$(cut -d' ' -f1 /proc/loadavg)"
awk -v l="$load" 'BEGIN { exit !(l < 0.5) }' && ok "load average $load" || warn "load average $load: wait for idle, a busy CPU contaminates latency"

gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
if [[ -r "$gov" ]]; then ok "cpu governor $(cat "$gov") (record it with the results)"; else warn "cpu governor not readable"; fi

if command -v vcgencmd > /dev/null; then
  temp="$(vcgencmd measure_temp | tr -dc '0-9.')"
  awk -v t="$temp" -v h="$HOT_C" 'BEGIN { exit !(t < h) }' && ok "soc temp ${temp} C" || fail "soc temp ${temp} C at idle: fix cooling first"
  throttled="$(vcgencmd get_throttled | cut -d= -f2)"
  # Bits 0-3: happening now; bits 16-19: has happened since boot. Under-voltage (bits 0, 16) usually means a weak PSU.
  [[ "$throttled" == "0x0" ]] && ok "throttled 0x0" || fail "throttled $throttled: under-voltage or heat since boot; use the 27 W PSU and an active cooler, then reboot"
  supply="$(vcgencmd pmic_read_adc EXT5V_V 2>/dev/null | tail -1)"
  [[ -n "$supply" ]] && ok "supply: $supply"
else
  warn "vcgencmd not found: temperature and throttling cannot be recorded"
fi
fan="$(cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input 2>/dev/null | head -1)"
[[ -n "$fan" ]] && ok "fan ${fan} rpm" || warn "no fan reading: note the cooler in the log by hand"

command -v podman > /dev/null && ok "podman $(podman --version | cut -d' ' -f3)" || warn "podman missing: needed to build the runner and the wheel"
command -v uv > /dev/null && ok "uv present" || warn "uv missing: needed for the Python 3.12 venv"

# Files: the same bytes as the dev box, or the numbers describe a different model.
check_sha() {
  local path="$ROOT/$1" want="$2"
  [[ -s "$path" ]] || { fail "missing $1"; return; }
  [[ "$(sha256sum "$path" | cut -d' ' -f1)" == "$want" ]] && ok "sha256 $1" || fail "sha256 differs: $1"
}
check_sha artifacts/llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte 8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b
check_sha artifacts/llama/tokenizer.model 82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55
check_sha runs/v6_final/export/tomato_effnet_b0_t2_fp32_xnnpack.pte 430b584d133af6f8d42d7e7a9e6d77c568d509ed045e662a563cb448b42b5445
for f in runs/v6_final/export/decision.json runs/v6_final/golden.json scripts/g1/prompt.txt scripts/g2/bench_matrix.json; do
  [[ -s "$ROOT/$f" ]] && ok "present $f" || fail "missing $f"
done
# On the dev box these are symlinks into data/: -L follows them, and a dangling link (plain copy) is not counted.
images="$(find -L "$ROOT/runs/bench/images" -type f 2>/dev/null | wc -l)"
(( images >= 30 )) && ok "bench images $images" || fail "bench images $images, expected 30 (copy with rsync -L; see README)"
[[ -e "$ROOT/data/sealed_test.sha256" ]] && warn "sealed split listing is on this board: sealed images are not needed on the Pi for G2"

echo; (( fails == 0 )) && echo "PREFLIGHT OK" || echo "PREFLIGHT: $fails FAIL"
exit $(( fails > 0 ))
