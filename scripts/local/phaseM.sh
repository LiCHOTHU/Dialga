#!/usr/bin/env bash
# PHASE M: the recon-vs-factorization FRONTIER.
#
# Phase I (3 seeds, 60 ep) showed the "objective fix" buys -21% recon by WEAKENING the
# split: deleting z_static drops from costing 11% to 6%, wrong-video from 21% to 10%,
# and z_dyn's dominance nearly doubles. lambda_indep is the dial between the two goals.
#
# Target: recon clearly below the 60-epoch baseline (0.0089) while zs_cost AND
# wrong-video stay at or above baseline (11% / 21%). Sweep the dial with the teacher
# holding the split up, 3 seeds, because the single-seed rankings proved unreliable
# (attr mAP seed-sd is +-0.005-0.018, larger than most gaps I ranked by).
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/frontier; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --d_static 96 --static_grid 4 --epochs 60 --lambda_consist 3 \
   --static_target video_median"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
for S in 0 1 2; do
  run M1_i03_t1_s$S --seed $S --lambda_indep 0.3 --lambda_static_tgt 1.0
  run M2_i03_t2_s$S --seed $S --lambda_indep 0.3 --lambda_static_tgt 2.0
  run M3_i1_t2_s$S  --seed $S --lambda_indep 1.0 --lambda_static_tgt 2.0
  run M4_i01_t2_s$S --seed $S --lambda_indep 0.1 --lambda_static_tgt 2.0
done
echo "PHASEM_DONE"
