#!/usr/bin/env bash
# PHASE J: the missing cross. Every OBJECTIVE result so far is CLEVRER-only and every
# ARCHITECTURE result is SSv2-only, so the two winners have never met.
#
#   objective winner (CLEVRER): lambda_indep 0, lambda_consist 3, median teacher
#                               -17% recon, +0.025 mAP, +0.081 zs-zd
#   architecture winner (SSv2): video-level z_static at 384/8x8
#                               -9.4% recon, 3 seeds, 6.5 pooled SDs
#
# 2x2 on SSv2: {today's objective, winning objective} x {per-chunk, video-level}.
# If they are additive, the corner arm is the model to commit to.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/cross; mkdir -p "$OUT"
B="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
   --epochs 30 --batch_size 16 --preload --max_videos 2000 --eval_every 10 --seed 0"
OBJ="--lambda_indep 0 --lambda_consist 3 --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
# --- 2x2 ---
run J1_old_perchunk  --d_static 96  --static_grid 4 --mem_update none
run J2_old_video     --d_static 384 --static_grid 8 --mem_update video
run J3_new_perchunk  --d_static 96  --static_grid 4 --mem_update none  $OBJ
run J4_new_video     --d_static 384 --static_grid 8 --mem_update video $OBJ
# --- and the winning objective at the winning rate, per-chunk (rate is free here) ---
run J5_new_perchunk_r384 --d_static 384 --static_grid 8 --mem_update none $OBJ
echo "PHASEJ_DONE"
