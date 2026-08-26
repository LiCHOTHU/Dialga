#!/usr/bin/env bash
# Overnight search for a z_static that is (a) more semantic and (b) reconstructs better.
#
# Run on CLEVRER, because it is the ONLY local dataset that can measure semantics:
# ground-truth attributes and per-object speeds give an informative probe, whereas the
# SSv2 action probe sits at chance (0.02-0.04 vs 0.020) and cannot answer anything.
# Reconstruction generality gets re-checked on SSv2 at the end for the winner only.
#
# What is already known, so it is not re-litigated here:
#   * static rate is the binding constraint on the static target -- PCA keeps 68.3% of
#     it at 96 floats, 90.7% at 384, 97.1% at 768
#   * grid resolution alone is NOT free: grid 4->8 at ~fixed rate (H7) made recon WORSE
#     (+2.2%) and gutted zs_cost (0.52 -> 0.22), because 128 floats on an 8x8 grid is
#     only 2 channels per cell. Channels and resolution trade off, so PHASE B sweeps
#     the (channels x grid) shape at MATCHED rate rather than assuming spatial is free
#   * teacher and video-level sharing are SUBSTITUTES: the teacher helps a per-chunk
#     code (-5.3%) and hurts a video-level one (+3.3%), so they are swept separately
#   * InfoNCE runs with batch-1 negatives only: at batch 16 that is 15 negatives, which
#     is very thin for the one term that pushes identity into z_static. PHASE C tests it
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/overnight; mkdir -p "$OUT"
BASE="--cache_dir outputs/cache/clevrer_W33_10k --epochs 30 --preload \
      --max_videos 2000 --eval_every 10 --seed 0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $BASE --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }

# ---- PHASE B: rate x shape. Does spending on z_static buy SEMANTICS as well as recon?
# At each rate, try the shapes that divide the grid area, so channels-vs-resolution is
# swept rather than assumed.
echo "########## PHASE B: rate x shape ##########"
run B_r96_g4    --batch_size 16 --d_static 96  --static_grid 4 --mem_update none   # today
run B_r192_g4   --batch_size 16 --d_static 192 --static_grid 4 --mem_update none
run B_r384_g4   --batch_size 16 --d_static 384 --static_grid 4 --mem_update none
run B_r384_g8   --batch_size 16 --d_static 384 --static_grid 8 --mem_update none
run B_r768_g8   --batch_size 16 --d_static 768 --static_grid 8 --mem_update none
run B_r768_g4   --batch_size 16 --d_static 768 --static_grid 4 --mem_update none

# ---- PHASE C: the objective. Which loss term actually buys semantics?
# InfoNCE is the only self-supervised term pushing identity into z_static and it runs
# on in-batch negatives alone, so batch size IS the negative count.
echo "########## PHASE C: objective ##########"
BEST="--d_static 384 --static_grid 8 --mem_update none"
run C_nce_b64    --batch_size 64 $BEST                       # 63 negatives, not 15
run C_nce_b8     --batch_size 8  $BEST                       # 7  negatives (control)
run C_nce_off    --batch_size 16 $BEST --lambda_consist 0.0  # is InfoNCE load-bearing?
run C_indep_off  --batch_size 16 $BEST --lambda_indep 0.0
run C_indep_3    --batch_size 16 $BEST --lambda_indep 3.0
run C_tgt_median --batch_size 16 $BEST --static_target video_median --lambda_static_tgt 1.0
run C_tgt_mean   --batch_size 16 $BEST --static_target video_mean   --lambda_static_tgt 1.0

# ---- PHASE D: memory, at the shape/objective the phases above pick out
echo "########## PHASE D: memory ##########"
run D_video      --batch_size 16 $BEST --mem_update video
run D_patchvideo --batch_size 16 $BEST --mem_update patch_video
run D_patch_tgt  --batch_size 16 $BEST --mem_update patch \
                 --static_target video_median --lambda_static_tgt 1.0
echo "OVERNIGHT_DONE"
