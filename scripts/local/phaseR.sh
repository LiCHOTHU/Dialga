#!/usr/bin/env bash
# PHASE R: FIX THE RATE ALLOCATION. Today's split is inverted relative to where the
# information lives.
#
#   measured energy:   91.5% static / 8.5% per-frame residual
#   today's bits:       4.0% static / 96.0% dynamic
#
# PCA on each target gives the best linear code at each budget, and predicts that at
# an unchanged ~2400-float budget, moving bits INTO z_static cuts total error 6-9x:
#     (96,256)=2400 -> 0.314   (today)
#     (768,192)=2496 -> 0.054
#    (1536,128)=2688 -> 0.035
#     (768,128)=1920 -> 0.059  at 20% LESS total rate than today
# R tests whether a trained model can realise that, on both decoders -- the grid one
# is free to ignore z_static (and has, at every rate so far), while base+delta forces
# the time-constant part through it.
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
# grid decoder: does the model USE a bigger z_static if given one?
run R1_s768_d192   --d_static 768  --d_dyn 192
run R2_s1536_d128  --d_static 1536 --d_dyn 128
run R3_s768_d128   --d_static 768  --d_dyn 128
run R4_s1536_d64   --d_static 1536 --d_dyn 64
# base+delta: same allocations, with the split enforced
run R5_bd_s768_d192  --decoder basedelta --d_static 768  --d_dyn 192
run R6_bd_s1536_d128 --decoder basedelta --d_static 1536 --d_dyn 128
run R7_bd_s768_d128  --decoder basedelta --d_static 768  --d_dyn 128
echo "PHASER_DONE"
