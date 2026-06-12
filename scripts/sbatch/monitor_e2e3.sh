#!/usr/bin/env bash
# monitor_e2e3.sh — keep the v5.1.2 VAE-unfreeze runs (e2, e3) alive.
#
# Loop: every SLEEP_INTERVAL_MIN minutes, for each monitored experiment
#   • If it is currently in the queue (PENDING / RUNNING / REQUEUED), skip.
#   • Else, if its out_dir already has a DONE marker, it finished — skip.
#   • Else, resubmit the matching sbatch. The trainer's `--resume auto` picks up
#     from <out_dir>/last.pt; the first launch (empty out_dir) starts fresh.
#
# This REPLACES the in-sbatch self-chaining (now removed from the e2/e3 sbatch).
# The trainer writes <out_dir>/DONE on natural completion at epoch 200, which is
# the single source of truth for "done".
#
# IMPORTANT: SLURM job names must be ≤7 chars and match JOBS below exactly, so
# the squeue `%j` lookup can never be truncated into a mismatch (which would
# double-submit). e2 -> e2_enc, e3 -> e3_dec.
#
# Env overrides (defaults shown):
#   SLEEP_INTERVAL_MIN=10      # minutes between sweeps
#   SLEEP_BETWEEN_SUBMITS=2    # seconds to pause after each submit
#
# Usage (run inside tmux/screen so it survives logout):
#   tmux new -s mon
#   bash scripts/sbatch/monitor_e2e3.sh
#
# Stop with Ctrl-C. Idempotent — re-running is harmless.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-10}"
SLEEP_BETWEEN_SUBMITS="${SLEEP_BETWEEN_SUBMITS:-2}"
OUT_BASE="/storage/scratch1/8/lwang831/dialga_outputs"

# Each row: <slurm_job_name>|<sbatch_file_basename>|<out_dir_subdir>
# job_name MUST equal the sbatch's #SBATCH --job-name (≤7 chars).
# out_dir_subdir MUST equal the sbatch's RUN_TAG.
JOBS=(
    "e2_enc|v512_e2_enc_unfrozen.sbatch|v512_e2_enc_unfrozen"
    "e3_dec|v512_e3_dec_unfrozen.sbatch|v512_e3_dec_unfrozen"
)

echo "▶ Monitoring v5.1.2 VAE-unfreeze jobs"
echo "   user:      ${USER}"
echo "   sbatch:    ${SCRIPT_DIR}"
echo "   out_base:  ${OUT_BASE}"
echo "   interval:  ${SLEEP_INTERVAL_MIN}m"
echo "   jobs:"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r job_name sbatch_file out_subdir <<< "${entry}"
    echo "     - ${job_name}  → ${out_subdir}"
done
echo

while true; do
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    active="$(squeue -u "${USER}" -h -o "%j" 2>/dev/null || true)"

    echo "============== ${now} =============="
    for entry in "${JOBS[@]}"; do
        IFS='|' read -r job_name sbatch_file out_subdir <<< "${entry}"

        # 1. Already in queue (PENDING / RUNNING / REQUEUED)?
        if printf '%s\n' "${active}" | awk -v j="${job_name}" '$0==j {found=1} END {exit !found}'; then
            echo "[OK]   ${job_name} active in queue"
            continue
        fi

        # 2. DONE marker already on disk?
        out_dir="${OUT_BASE}/${out_subdir}"
        if [[ -f "${out_dir}/DONE" ]]; then
            echo "[DONE] ${job_name} ($(cat "${out_dir}/DONE" 2>/dev/null | tr -d '\n'))"
            continue
        fi

        # 3. Otherwise resubmit. --resume auto continues from last.pt; first
        #    launch (empty out_dir) starts fresh.
        sbatch_path="${SCRIPT_DIR}/${sbatch_file}"
        if [[ ! -f "${sbatch_path}" ]]; then
            echo "[ERR]  ${job_name} sbatch script missing: ${sbatch_path}"
            continue
        fi

        echo "[MISS] ${job_name} not in queue and no DONE — submitting"
        sbatch "${sbatch_path}" || echo "[ERR]  sbatch returned non-zero for ${job_name}"

        if (( SLEEP_BETWEEN_SUBMITS > 0 )); then
            sleep "${SLEEP_BETWEEN_SUBMITS}"
        fi
    done

    echo "Sleeping ${SLEEP_INTERVAL_MIN} minutes..."
    sleep "${SLEEP_INTERVAL_MIN}m"
done
