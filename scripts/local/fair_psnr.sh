#!/usr/bin/env bash
# APPLES-TO-APPLES. The baseline protocol freezes a representation and trains an
# equal-capacity head to decode it; measured that way the FULL 27648-float latent
# reaches only 27.72 dB, so the head -- not the representation -- is the ceiling, and
# our 34.63 dB (jointly trained encoder+decoder) is not comparable to it.
# This runs OUR frozen code through the identical head so the column means something.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
python -u scripts/probes/clevrer_decode_baselines.py \
  --ckpt outputs/final/BASE60_s0/ckpt.pt \
  --cache_dir outputs/cache/clevrer_W33_10k \
  --video_root datasets/CLEVRER/train_video \
  --methods ours random \
  --max_videos 1200 --epochs 60 --dec_hidden 384 --pixel --pixel_n 64 \
  --out outputs/logs/fair_psnr.json
echo FAIR_PSNR_DONE
