#!/usr/bin/env bash
set -uo pipefail
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
cd /home/licho/workspace/Dialga; export PYTHONPATH=/home/licho/workspace/Dialga
exec python -u scripts/cache_dino_patch.py \
  --wan_cache_dir outputs/cache/clevrer_W33_10k \
  --out_dir outputs/cache/dino_clevrer_W33 \
  --model facebook/dinov2-small --batch_windows 8 --num_workers 6
