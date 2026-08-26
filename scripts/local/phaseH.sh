#!/usr/bin/env bash
# PHASE H: combine the two complementary recipes found tonight.
#
#   E5  (indep 0, nce 2, median teacher)   recon -17%, mAP +0.013, but still-gap DOWN
#   F6  (DINO -> z_static only, lambda .1) mAP +0.020, still-gap HELD, recon unchanged
#
# and the routing result they sit on: sending the DINOv2 target to BOTH codes (what
# train_v5's AuxSemanticDecoder does today) collapses the stationary/moving gap from
# +0.065 to +0.004 and costs 15% recon, because the signal takes the cheap path
# through z_dyn -- which lambda_indep is simultaneously trying to strip of identity.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/overnight; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 10 \
   --seed 0 --batch_size 16 --d_static 96 --static_grid 4 \
   --dino_cache_dir outputs/cache/dino_clevrer_W33 --dino_to static"
MED="--static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
run H1_e5_dino01  --epochs 30 --lambda_indep 0 --lambda_consist 2 $MED --lambda_dino 0.1
run H2_e5_dino05  --epochs 30 --lambda_indep 0 --lambda_consist 2 $MED --lambda_dino 0.5
run H3_i0n2_dino01 --epochs 30 --lambda_indep 0 --lambda_consist 2 --lambda_dino 0.1
run H4_e5_dino01_t05 --epochs 30 --lambda_indep 0 --lambda_consist 2 \
                     --static_target video_median --lambda_static_tgt 0.5 --lambda_dino 0.1
run H5_i03n2_dino01 --epochs 30 --lambda_indep 0.3 --lambda_consist 2 $MED --lambda_dino 0.1
run H6_best_e60   --epochs 60 --lambda_indep 0 --lambda_consist 2 $MED --lambda_dino 0.1
echo "PHASEH_DONE"
