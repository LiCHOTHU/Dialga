#!/usr/bin/env bash
# PHASE S: BALANCED codes -- z_static total == z_dyn total, with z_dyn kept small.
#
# z_dyn is paid ONCE PER FRAME, so equal totals means d_static = 9 x d_dyn/frame.
# Measured energy is 91.5% static / 8.5% residual, so a balanced split is not
# arbitrary -- it is much closer to where the information actually is than today's
# 4%/96%. PCA predicts, at or below today's rate:
#     ( 576, 64) = 1152 floats  24.0x  50/50  pred err 0.090
#     (1152,128) = 2304 floats  12.0x  50/50  pred err 0.045   (4% BELOW today's rate)
#     (2304,256) = 4608 floats   6.0x  50/50  pred err 0.025
#     (  96,256) = 2400 floats  11.5x   4/96  pred err 0.314   <- today
# base+delta arms make z_static EXPLICIT: the time-constant part of the
# reconstruction is produced by z_static alone and is unreachable from z_dyn.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/alloc; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0 --static_grid 8"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
run S1_bal_1152_128     --d_static 1152 --d_dyn 128                      # 2304f, 50/50
run S2_bal_1152_128_bd  --d_static 1152 --d_dyn 128 --decoder basedelta  #   explicit
run S3_bal_576_64       --d_static 576  --d_dyn 64                       # 1152f, 24x
run S4_bal_576_64_bd    --d_static 576  --d_dyn 64  --decoder basedelta
run S5_bal_2304_256     --d_static 2304 --d_dyn 256                      # 4608f, 6x
run S6_bal_2304_256_bd  --d_static 2304 --d_dyn 256 --decoder basedelta
echo "PHASES_DONE"
