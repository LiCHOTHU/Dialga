#!/usr/bin/env bash
# PSNR on all 3 seeds and 4x the chunks: 48 chunks / 1 seed is too thin for a dB claim.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
CK=(); LB=()
for a in BASE60 WIN60 WIND60; do for s in 0 1 2; do
  [ -f outputs/final/${a}_s${s}/ckpt.pt ] && { CK+=(outputs/final/${a}_s${s}/ckpt.pt); LB+=(${a}_s${s}); }
done; done
python -u scripts/local/eval_psnr.py --ckpts "${CK[@]}" --labels "${LB[@]}" \
  --n_chunks 192 --out outputs/logs/psnr_full.json
