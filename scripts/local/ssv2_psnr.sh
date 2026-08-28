#!/usr/bin/env bash
# SSv2 matched-protocol PSNR table. Our method first so the headline row lands early,
# then the ceiling/trivial rows, then the RGB encoders (slow: they decode video).
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1   # VideoFlexTok trips inductor here
python -u scripts/local/ssv2_decode_baselines.py \
  --cache_dir outputs/cache/ssv2_W17_full \
  --video_dir datasets/ssv2/videos \
  --ckpt outputs/FINAL_SSV2/ckpt.pt \
  --methods ours wanflat wanmean videomae dinov2 videoflextok \
  --n_train 4000 --n_val 600 --n_pixel 96 --epochs 60 \
  --out outputs/logs/ssv2_decode.json
