#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
# the "ceiling" arm: probe the FULL raw Wan latent on the same split our model used
python -u scripts/probes/ssv2_action_probe.py \
  --ckpt outputs/FINAL_SSV2/ckpt.pt \
  --train_cache outputs/cache/ssv2_W17_full \
  --val_cache   outputs/cache/ssv2_W17_full 2>&1 | tail -25
echo CEILING_DONE
