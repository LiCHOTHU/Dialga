#!/usr/bin/env bash
# Does the semantic factorization SURVIVE these changes? SSv2 cannot answer it -- its
# action probe sits at chance (0.02-0.04 vs 0.020) with ~500 val clips over ~170
# classes. CLEVRER has ground-truth attributes AND per-object speeds, so it can:
#   attribute mAP from z_static vs z_dyn   (is identity still in the static code?)
#   stationary vs moving split             (the original intuition)
#   zs_cost / swap                         (is each code still necessary?)
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/semantic_check; mkdir -p "$OUT"
COMMON="--cache_dir outputs/cache/clevrer_W33_10k --epochs 30 --batch_size 16 \
        --preload --max_videos 2000 --eval_every 10 --seed 0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $COMMON --out_dir "$d" "$@" && touch "$d/DONE"; }
run R1_today       --d_static 96  --static_grid 4 --mem_update none
run R2_patch_tgt   --d_static 384 --static_grid 8 --mem_update patch \
                   --static_target video_median --lambda_static_tgt 1.0
run R3_patchvideo  --d_static 384 --static_grid 8 --mem_update patch_video
echo "SEMANTIC_DONE"
