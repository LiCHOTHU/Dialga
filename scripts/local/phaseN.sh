#!/usr/bin/env bash
# PHASE N: attack the actual reconstruction bottleneck.
#
# PCA on the cached latents (zero training, 1500 chunks):
#   energy split               91.5% static / 8.5% dynamic  (per-frame residual)
#   z_static @96 floats        keeps 68.3% of the static part -> 29.0% of total error
#   z_dyn    @256 floats/frame keeps 72.1% of the residual   ->  2.4% of total error
# So z_static's rate is ~12x the bottleneck z_dyn's is -- yet the trained models put
# nearly everything in z_dyn (deleting z_dyn costs 714-1450%, z_static 6-11%). The
# model is paying NINE TIMES over for static content, once per frame.
#
# base+delta forces mean_t through z_static structurally. It hurt before ONLY because
# d_static was 96 (68% capture); at 384/768 it is 90.7%/97.1%. N pairs the constraint
# with the rate it always needed. Objective = the Phase-I winner.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/psnr_push; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
#     the pairing that was never tried: the structural constraint AT the rate it needs
run N1_bd_s768_d256  --decoder basedelta --d_static 768 --static_grid 8 --d_dyn 256
run N2_bd_s768_d512  --decoder basedelta --d_static 768 --static_grid 8 --d_dyn 512
run N3_bd_s384_d512  --decoder basedelta --d_static 384 --static_grid 8 --d_dyn 512
#     controls: same rate WITHOUT the constraint, and rate poured into z_dyn instead
run N4_grid_s768_d512 --d_static 768 --static_grid 8 --d_dyn 512
run N5_grid_s96_d1024 --d_static 96  --static_grid 4 --d_dyn 1024
run N6_grid_s768_d1024 --d_static 768 --static_grid 8 --d_dyn 1024
run N7_bd_s768_d1024 --decoder basedelta --d_static 768 --static_grid 8 --d_dyn 1024
echo "PHASEN_DONE"
