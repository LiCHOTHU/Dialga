#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SBATCH_SCRIPT="${SBATCH_SCRIPT:-${REPO_ROOT}/scripts/train_dialga.sbatch}"
JOB_NAME="${JOB_NAME:-dialga-train}"
SLEEP_INTERVAL_MIN="${SLEEP_INTERVAL_MIN:-5}"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "[error] sbatch script not found: ${SBATCH_SCRIPT}" >&2
  exit 1
fi

SBATCH_ARGS=("$@")

echo "▶ Monitoring DIALGA training job"
echo "   user:     ${USER}"
echo "   job:      ${JOB_NAME}"
echo "   sbatch:   ${SBATCH_SCRIPT}"
echo "   interval: ${SLEEP_INTERVAL_MIN}m"
if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
  echo "   args:     ${SBATCH_ARGS[*]}"
else
  echo "   args:     <none>"
fi
echo

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  active_jobs="$(squeue -u "${USER}" -h -o "%j" || true)"
  running_count="$(
    awk -v j="${JOB_NAME}" '$0==j {c++} END {print c+0}' <<< "${active_jobs}"
  )"

  echo "============== ${now} =============="
  if (( running_count > 0 )); then
    echo "[OK] ${JOB_NAME} active (${running_count} job(s) in queue)."
  else
    echo "[MISS] ${JOB_NAME} missing. Submitting..."
    sbatch --job-name="${JOB_NAME}" "${SBATCH_SCRIPT}" "${SBATCH_ARGS[@]}"
  fi

  echo "Sleeping ${SLEEP_INTERVAL_MIN} minutes..."
  sleep "${SLEEP_INTERVAL_MIN}m"
done
