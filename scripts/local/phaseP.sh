#!/usr/bin/env bash
# PHASE P: map the PSNR-vs-factorization frontier and find where 40 dB actually is.
#
# Measured so far (96 held-out chunks, ceiling 46.85 dB):
#   today       2400f  34.63 dB  zs_cost  11%
#   bd  9984f   9984f  34.37 dB  zs_cost 823%   <- factorization, no dB
#   grid 9984f  9984f  38.40 dB  zs_cost   3%   <- dB, no factorization
# The two goals are currently anti-correlated. P fills in the middle (partial
# constraint / intermediate rates) and pushes rate further to see where 40 dB lands.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/psnr_push
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
# where is 40 dB? push rate on the unconstrained decoder
run P1_grid_s768_d2048  --d_static 768 --static_grid 8 --d_dyn 2048
run P2_grid_s768_d1536  --d_static 768 --static_grid 8 --d_dyn 1536
# the middle: keep the teacher pressure but NOT the hard base+delta constraint,
# at the rate where the grid decoder starts to pay
run P3_grid_s768_d1024_t4 --d_static 768 --static_grid 8 --d_dyn 1024 --lambda_static_tgt 4.0
run P4_grid_s768_d1024_i03 --d_static 768 --static_grid 8 --d_dyn 1024 --lambda_indep 0.3
echo "PHASEP_DONE"
