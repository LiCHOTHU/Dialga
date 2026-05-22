#!/usr/bin/env bash
# ===================================================================
# Submit the Iter 22 three-job pipeline:
#
#   Job 1 (cache) — smoke + full DINO cache. No deps. ~5h max.
#   Job 2 (22a)   — Iter 22a training + probes. depends on Job 1. ~4h.
#   Job 3 (22c)   — Iter 22c training + probes. NO dep on Job 1.   ~4h.
#
# Wallclock (with two free GPUs): max(cache + 22a, 22c) ≈ 6-7 h.
# Wallclock (with one GPU): all three queued, ~10-12 h.
#
# All three jobs share a single STAMP env var so the output dirs
# and log lines line up.
#
# Usage:
#   bash scripts/submit_iter22.sh
# ===================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
export STAMP

echo "STAMP=${STAMP}"

JID_CACHE=$(sbatch --parsable --export=ALL,STAMP="${STAMP}" \
    scripts/sbatch_iter22_cache.sh)
echo "submitted cache  job: ${JID_CACHE}"

JID_22A=$(sbatch --parsable --dependency=afterok:"${JID_CACHE}" \
    --export=ALL,STAMP="${STAMP}" \
    scripts/sbatch_iter22a_train.sh)
echo "submitted 22a    job: ${JID_22A}  (depends on ${JID_CACHE})"

JID_22C=$(sbatch --parsable --export=ALL,STAMP="${STAMP}" \
    scripts/sbatch_iter22c_train.sh)
echo "submitted 22c    job: ${JID_22C}  (independent)"

echo ""
echo "Track progress:"
echo "  squeue -u \$USER"
echo "  tail -f /storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/dialga_iter22_cache_${JID_CACHE}.out"
echo "  tail -f /storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/dialga_iter22a_${JID_22A}.out"
echo "  tail -f /storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/dialga_iter22c_${JID_22C}.out"
echo ""
echo "When done, results land in:"
echo "  outputs/iter22a_dino_${STAMP}/  (probe_identity_linear.json, probe_identity_diag.json)"
echo "  outputs/iter22c_attrs_${STAMP}/ (probe_identity_linear.json, probe_identity_diag.json)"
