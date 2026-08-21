#!/usr/bin/env bash
# Watchdog: for each DIALGA CLEVRER chain, if it has NO job in the queue
# (neither RUNNING nor PENDING) and is not marked DONE, relaunch its sbatch
# with --resume auto (ATTEMPT=1, fresh chain budget). Idempotent and safe to
# run on a schedule — it only SUBMITS, never cancels.
set -uo pipefail

USER=lwang831
SBATCH_DIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga/scripts/sbatch"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"

# tag -> sbatch file (only chains we want kept alive). 2026-06-21: the entire
# v512/v513/v514/v53/v54 CLEVRER fleet was deliberately cancelled (too many
# experiments); ONLY the v5.5 budget-matched pool ablation pair is live now.
# Do NOT re-add the old tags here or the watchdog will resurrect cancelled runs.
TAGS=(
  v55_pool_mean v55_pool_slot v55_slot_gaze v55_pixel_only
  v55_mean_pixfull v55_mean_pixhi v55_slot_pixfull
)

ts() { date "+%Y-%m-%d %H:%M:%S"; }
echo "[$(ts)] watchdog scan ----------------------------------------"
LIVE="$(squeue -u "$USER" -h -o '%j' 2>/dev/null)"

for tag in "${TAGS[@]}"; do
  n=$(echo "$LIVE" | grep -c "^${tag}$")
  if [ -f "${CEDAR}/${tag}/DONE" ]; then
    echo "  ${tag}: DONE — skip"
    continue
  fi
  if [ "$n" -ge 1 ]; then
    echo "  ${tag}: alive (${n} job/s)"
    continue
  fi
  sb="${SBATCH_DIR}/${tag}.sbatch"
  if [ ! -f "$sb" ]; then
    echo "  ${tag}: LAPSED but no sbatch at ${sb} — SKIP"
    continue
  fi
  jid=$(sbatch --export="ALL,ATTEMPT=1" "$sb" 2>&1 | awk '{print $NF}')
  echo "  ${tag}: LAPSED -> relaunched (resume auto) job ${jid}"
done
echo "[$(ts)] watchdog scan done."
