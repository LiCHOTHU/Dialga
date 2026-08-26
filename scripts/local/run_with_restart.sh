#!/usr/bin/env bash
# Auto-restart wrapper: local background python jobs get silently SIGKILL'd on this
# box (see memory feedback_background_jobs). Relaunch the inner command until it
# prints its completion sentinel or we hit MAX_TRIES. Inner script MUST be resumable.
#   usage: run_with_restart.sh <sentinel> <logfile> <cmd...>
set -uo pipefail
SENTINEL="$1"; LOG="$2"; shift 2
MAX_TRIES="${MAX_TRIES:-200}"
for i in $(seq 1 "$MAX_TRIES"); do
  echo "=== attempt $i @ $(date -Is) ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  rc=$?
  if grep -q "$SENTINEL" "$LOG"; then
    echo "=== SENTINEL '$SENTINEL' found; done @ $(date -Is) ===" >> "$LOG"; exit 0
  fi
  echo "=== attempt $i exited rc=$rc without sentinel; retrying in 10s ===" >> "$LOG"
  sleep 10
done
echo "=== gave up after $MAX_TRIES attempts ===" >> "$LOG"; exit 1
