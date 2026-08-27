#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
while ! grep -q CEILING_OK outputs/logs/ceiling2.log 2>/dev/null; do sleep 60; done
# seed 1 and 2 of the committed config on full SSv2 -- the headline numbers are
# currently single-seed at this scale
for S in 1 2; do
  d=outputs/FINAL_SSV2_s$S; [ -f "$d/DONE" ] && continue
  python -u scripts/local/train_memory.py --dataset ssv2 --chunk_size_lat 5 \
    --cache_dir outputs/cache/ssv2_W17_full --out_dir "$d" \
    --epochs 60 --batch_size 16 --eval_every 20 --seed $S --preload \
    --decoder basedelta --static_grid 8 --d_static 576 --d_dyn 64 \
    --lambda_indep 0 --lambda_consist 3 \
    --static_target video_median --lambda_static_tgt 1.0 && touch "$d/DONE"
done
echo SSV2_SEEDS_OK
