#!/usr/bin/env bash
# PHASE T: STARVE z_dyn. Measured directly (decode from one code alone, in dB):
#
#   model                   full    z_static ONLY   z_dyn ONLY
#   today (96f static)     34.95         24.90dB      34.42dB
#   grid  (768f static)    38.06         24.89dB      37.88dB   <- 8x the static rate
#   base+delta (768f)      34.83         28.63dB      25.65dB      changed NOTHING
#
# Giving z_static 8x more rate moved its solo reconstruction by 0.01 dB and left 76%
# of its dimensions unused. A model with a large z_dyn will not use z_static, whatever
# rate it is given. Two ways to force the issue, tested together here:
#   (a) shrink z_dyn so it CANNOT hold the scene
#   (b) base+delta so the time-constant part is unreachable from z_dyn
# Also compresses z_dyn SPATIALLY (dyn_grid 8 -> 4), untested so far -- v5.9 showed a
# spatial z_dyn is needed for real video, but never asked how fine that grid must be.
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
#  z_dyn starved to 1 channel: 1152 + 9x64 = 1728 floats, 16.0x, z_dyn only 33% of code
run T1_s1152_d64      --d_static 1152 --d_dyn 64
run T2_s1152_d64_bd   --d_static 1152 --d_dyn 64  --decoder basedelta
#  z_dyn starved AND spatially coarser (4x4 instead of 8x8)
run T3_s1152_d32g4_bd --d_static 1152 --d_dyn 32  --dyn_grid 4 --decoder basedelta
run T4_s1152_d128g4_bd --d_static 1152 --d_dyn 128 --dyn_grid 4 --decoder basedelta
#  static-dominant: 2304 + 9x64 = 2880 floats, z_dyn only 20% of the code
run T5_s2304_d64_bd   --d_static 2304 --d_dyn 64  --decoder basedelta
run T6_s2304_d64      --d_static 2304 --d_dyn 64
echo "PHASET_DONE"
