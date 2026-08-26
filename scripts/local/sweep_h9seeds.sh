#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
        --epochs 30 --batch_size 16 --preload --max_videos 2000 --eval_every 10 \
        --d_static 384 --static_grid 8"
for S in 1 2; do
  d=outputs/seed_sweep/H9_s$S; [ -f "$d/DONE" ] && continue
  echo "=================== ARM H9_s$S ==================="
  python -u scripts/local/train_memory.py $COMMON --out_dir "$d" --seed $S \
     --mem_update patch --static_target video_median --lambda_static_tgt 1.0 \
     && touch "$d/DONE"
done
echo "H9SEEDS_DONE"
