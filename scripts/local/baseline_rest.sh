#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
# VideoFlexTok trips torch.compile/inductor on this Blackwell + torch 2.7 stack
# (PassManager::run failed); the cluster sbatch scripts set this for the same reason.
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
python -u scripts/probes/clevrer_decode_baselines.py \
  --ckpt outputs/FINAL/F_cand_s0/ckpt.pt \
  --cache_dir outputs/cache/clevrer_W33_10k \
  --video_root datasets/CLEVRER/train_video \
  --methods videoflextok \
  --max_videos 1200 --epochs 60 --dec_hidden 384 --pixel --pixel_n 64 \
  --out outputs/logs/videoflextok_psnr.json
