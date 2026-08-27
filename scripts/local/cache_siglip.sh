#!/usr/bin/env bash
set -uo pipefail
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
cd /home/licho/workspace/Dialga; export PYTHONPATH=/home/licho/workspace/Dialga
exec python -u scripts/local/cache_siglip_patch.py \
  --wan_cache_dir outputs/cache/clevrer_W33_10k \
  --out_dir outputs/cache/siglip_clevrer_W33 \
  --model google/siglip-base-patch16-224 --batch_windows 8 --num_workers 6
