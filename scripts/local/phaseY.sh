#!/usr/bin/env bash
# PHASE Y: does a DECOMPOSED teacher reduce OVERLAP?
#
# Measured RBF-CKA between the codes, every config so far: 0.32 - 0.51. Nothing built
# tonight moved it, and lambda_indep is blind to it (reads ~0.003 for all of them,
# a 1.6x spread against a 60% spread in real overlap).
#
# The cause: z_static has a teacher, z_dyn has none, so nothing pushes their CONTENTS
# apart -- only the hinge pushes both to be necessary, which produced synergy without
# separation. Here each code is taught the half of the frame-wise semantic signal the
# other is explicitly not taught:
#     z_static <- median_t f_t          (what persists)
#     z_dyn    <- |f_t - median_t f_t|  (per-frame deviation)
# Disjoint by construction. Y2/Y3 are the controls that isolate the decomposition from
# simply having a teacher at all.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/teach; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --dino_cache_dir outputs/cache/dino_clevrer_W33 \
   --preload --max_videos 2000 --eval_every 30 --batch_size 16 --epochs 60 --seed 0 \
   --lambda_indep 0 --lambda_consist 3 --static_grid 8 --d_static 1152 --d_dyn 128 \
   --decoder basedelta --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
run Y1_decomposed  --lambda_dino 0.5 --lambda_dyn_teach 0.5   # both halves taught
run Y2_static_only --lambda_dino 0.5                          # only z_static taught
run Y3_no_teacher                                             # neither (= V3)
run Y4_decomp_hi   --lambda_dino 1.0 --lambda_dyn_teach 1.0
echo "PHASEY_DONE"
