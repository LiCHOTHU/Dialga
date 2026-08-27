#!/usr/bin/env bash
# PHASE K/L: knobs never touched in 50+ arms.
#   K  optimisation -- lr has been 3e-4 constant with NO schedule in every run tonight
#   L  rate balance -- z_dyn is 2304 floats against z_static's 96, a 24:1 split that
#      has never been swept. Shrinking z_dyn both cuts total rate AND forces content
#      into z_static, which is the whole point of the factorization.
# All on the winning objective (indep 0, nce 3, median teacher) at today's shape.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/tune2; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 20 \
   --batch_size 16 --d_static 96 --static_grid 4 --seed 0 \
   --lambda_indep 0 --lambda_consist 3 --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
# ---- K: optimisation
run K1_lr3e4_e60      --epochs 60 --lr 3e-4                              # control
run K2_lr1e3_e60      --epochs 60 --lr 1e-3
run K3_lr1e4_e60      --epochs 60 --lr 1e-4
run K4_cos3e4_e60     --epochs 60 --lr 3e-4 --lr_schedule cosine --warmup 3
run K5_cos1e3_e60     --epochs 60 --lr 1e-3 --lr_schedule cosine --warmup 3
run K6_cos1e3_e120    --epochs 120 --lr 1e-3 --lr_schedule cosine --warmup 5
# ---- L: rate balance (z_dyn per frame; 256 is today)
run L1_dyn128         --epochs 60 --lr 1e-3 --lr_schedule cosine --warmup 3 --d_dyn 128
run L2_dyn64          --epochs 60 --lr 1e-3 --lr_schedule cosine --warmup 3 --d_dyn 64
run L3_dyn512         --epochs 60 --lr 1e-3 --lr_schedule cosine --warmup 3 --d_dyn 512
echo "PHASEK_DONE"
