#!/usr/bin/env bash
# Combine the two winning mechanisms and tune what was never tuned.
#
# patch_video = MosaicMem retrieve-and-compose read ONCE per clip with a learned query
# set, so it gets the rate saving too: 384 static floats/clip (5504 total), the same
# budget as H2's attention-pooled video code and 21% below H9's per-chunk patch code.
# Q1/Q2 are the head-to-head against H2 (0.0731) at matched rate.
# lambda_static_tgt was fixed at 1.0 from the first smoke and never swept; Q3/Q4 do that.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/seed_sweep
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
        --epochs 30 --batch_size 16 --preload --max_videos 2000 --eval_every 10 \
        --d_static 384 --static_grid 8"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $COMMON --out_dir "$d" "$@" && touch "$d/DONE"; }
run Q1_pvideo_s0     --seed 0 --mem_update patch_video
run Q2_pvideo_tgt_s0 --seed 0 --mem_update patch_video --static_target video_median --lambda_static_tgt 1.0
run Q3_pvideo_t03_s0 --seed 0 --mem_update patch_video --static_target video_median --lambda_static_tgt 0.3
run Q4_pvideo_t3_s0  --seed 0 --mem_update patch_video --static_target video_median --lambda_static_tgt 3.0
echo "TUNE_DONE"
