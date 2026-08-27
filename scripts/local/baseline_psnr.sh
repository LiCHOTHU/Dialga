#!/usr/bin/env bash
# The missing number: PIXEL PSNR for VideoMAE / VideoFlexTok / DINOv2 / full-latent,
# each frozen and decoded back to the Wan latent by an EQUAL-CAPACITY head, then
# Wan-decoded to pixels. Without this there is no answer to "are we competitive".
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
python -u scripts/probes/clevrer_decode_baselines.py \
  --ckpt outputs/FINAL/F_cand_s0/ckpt.pt \
  --cache_dir outputs/cache/clevrer_W33_10k \
  --video_root datasets/CLEVRER/train_video \
  --methods ours \
  --max_videos 1200 --epochs 60 --dec_hidden 384 \
  --pixel --pixel_n 64 \
  --out outputs/logs/matched_committed.json
echo BASELINE_PSNR_DONE
