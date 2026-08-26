#!/usr/bin/env bash
# PHASE G: refine around E5, the first config that beats today's model on BOTH axes.
#
#   E5 = lambda_indep 0, lambda_consist 2, median teacher, 96 floats on a 4x4 grid
#        recon 0.0086 (-17%), attr mAP 0.757 (+0.014), zs-zd gap +0.069 (+0.017)
#
# The trade-off B/C found (load-bearing XOR semantic) is broken by combining three
# things that each repair what the others cost: dropping lambda_indep frees
# reconstruction, the median teacher restores the static/dynamic split, and a heavier
# InfoNCE WEIGHT (not more negatives -- batch 64 hurt) restores semantics.
# G walks the neighbourhood of that point and checks it survives longer training.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/overnight; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 \
   --eval_every 10 --seed 0 --batch_size 16 --d_static 96 --static_grid 4 \
   --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
run G1_i03_nce2   --epochs 30 --lambda_indep 0.3 --lambda_consist 2.0
run G2_i01_nce2   --epochs 30 --lambda_indep 0.1 --lambda_consist 2.0
run G3_i0_nce3    --epochs 30 --lambda_indep 0.0 --lambda_consist 3.0
run G4_i0_nce2_t05 --epochs 30 --lambda_indep 0.0 --lambda_consist 2.0 --lambda_static_tgt 0.5
run G5_i0_nce2_t2 --epochs 30 --lambda_indep 0.0 --lambda_consist 2.0 --lambda_static_tgt 2.0
run G6_i03_nce4   --epochs 30 --lambda_indep 0.3 --lambda_consist 4.0
run G7_i0_nce2_r192 --epochs 30 --lambda_indep 0.0 --lambda_consist 2.0 --d_static 192
run G8_i0_nce2_e60 --epochs 60 --lambda_indep 0.0 --lambda_consist 2.0
echo "PHASEG_DONE"
