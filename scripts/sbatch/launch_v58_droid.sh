#!/usr/bin/env bash
# Launch the v5.8 DROID moving-camera pipeline as a SLURM dependency chain:
#   extract  ->  (train ON  ||  train BLIND)  ->  readout
# The two train arms run in parallel; readout waits on BOTH (afterok).
# Usage:  bash scripts/sbatch/launch_v58_droid.sh   [N_EPISODES]
set -eo pipefail
SB="/storage/home/hcoda1/8/lwang831/workspace/Dialga/scripts/sbatch"
N_EPISODES="${1:-300}"

EXTRACT=$(sbatch --parsable --export=ALL,N_EPISODES="${N_EPISODES}" \
          "${SB}/v58_droid_extract.sbatch")
echo "extract      : ${EXTRACT}  (first ${N_EPISODES} episodes)"

ON=$(sbatch --parsable --dependency=afterok:${EXTRACT} \
     --export=ALL,ARM=on "${SB}/v58_droid_train.sbatch")
echo "train ON     : ${ON}  (afterok:${EXTRACT})"

BLIND=$(sbatch --parsable --dependency=afterok:${EXTRACT} \
        --export=ALL,ARM=blind "${SB}/v58_droid_train.sbatch")
echo "train BLIND  : ${BLIND}  (afterok:${EXTRACT})"

READOUT=$(sbatch --parsable --dependency=afterok:${ON}:${BLIND} \
          "${SB}/v58_droid_readout.sbatch")
echo "readout      : ${READOUT}  (afterok:${ON}:${BLIND})"

echo
echo "chain: ${EXTRACT} -> [${ON} || ${BLIND}] -> ${READOUT}"
