#!/usr/bin/env bash
set -uo pipefail
source /home/licho/anaconda3/etc/profile.d/conda.sh
conda activate dialga
cd /home/licho/workspace/Dialga
export PYTHONPATH=/home/licho/workspace/Dialga
exec python -u scripts/cache_wan_latents.py \
  --data_dir datasets/CLEVRER/train_video \
  --annotation_dir datasets/CLEVRER/annotations \
  --max_videos 10000 --window_length 33 \
  --deterministic_starts '0,32,64,95' \
  --out_dir outputs/cache/clevrer_W33_10k
