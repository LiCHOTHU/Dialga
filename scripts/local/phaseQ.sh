#!/usr/bin/env bash
# PHASE Q: z_dyn at the SAME channel depth as z_static (12ch on 8x8 = 768/frame).
# Symmetric codes: 768 static (once) + 768 x 9 dynamic = 7680 floats, 3.60x compression.
# Both decoders, so the factorization cost is measured at this point too.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/psnr_push
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0 --d_static 768 --static_grid 8 --d_dyn 768"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
run Q1_grid_12ch_sym
run Q2_bd_12ch_sym --decoder basedelta
echo "PHASEQ_DONE"
