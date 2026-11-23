#!/usr/bin/env bash
# Freeze a long job when the laptop is on battery and low, thaw it when mains is back (23 Nov, v10 run).
# SIGSTOP keeps the job in memory: nothing is lost and no result changes. Only signals the job's process group.
# Usage: setsid nohup scripts/power_guard.sh <driver pid> [pause at %] > logs/power_guard.log 2>&1 < /dev/null &
set -uo pipefail
PID=${1:?driver pid}; LOW=${2:-30}; EVERY=${EVERY:-10}
PS=${POWER_DIR:-/sys/class/power_supply}
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
[[ -n $PGID && $PGID == "$PID" ]] || { echo "pid $PID is not running or does not lead its own process group; refusing"; exit 1; }
say() { echo "$(date '+%F %T') $*"; }
on_ac() { grep -qs 1 "$PS"/A*/online; }
charge() { cat "$PS"/BAT*/capacity 2>/dev/null | head -1; }
paused=no
say "guarding group $PGID, pause at ${LOW}% on battery"
while kill -0 "$PID" 2>/dev/null; do
  c=$(charge)
  if [[ $paused == no ]] && ! on_ac && [[ -n $c && $c -le $LOW ]]; then
    kill -STOP -- "-$PGID" && paused=yes && say "PAUSED at ${c}%"
  elif [[ $paused == yes ]] && on_ac; then
    kill -CONT -- "-$PGID" && paused=no && say "RESUMED on mains at ${c}%"
  fi
  sleep "$EVERY"
done
say "job ended"
