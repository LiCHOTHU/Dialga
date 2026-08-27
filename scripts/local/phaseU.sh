#!/usr/bin/env bash
# PHASE U: COMPLEMENTARITY at scale. Neither code sufficient alone, both needed jointly.
#
# Why not orthogonality (the field-standard tool, used by Video-JEPA 2605.17165 and
# DiViD 2507.13934): measured here, L_indep sits at 0.0037 -- near-perfect
# decorrelation -- in a model whose z_static reconstructs 24.90 dB alone and is
# unchanged (24.89 dB) when given 8x the rate. A NOISE z_static satisfies
# orthogonality perfectly. It does not target usefulness at all.
#
# The hinge targets the measured quantity instead: each code ALONE must reconstruct at
# least (1+m)x worse than the pair. Smoke (300 vids, 6 ep) already gives zs_cost +372%
# / zd_cost +209% -- balanced and both necessary -- with the PLAIN grid decoder.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/comp; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0 --static_grid 8"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
#  balanced rate (1152 + 9x128 = 2304 floats, 12.0x, 50/50 split), margin sweep
run U1_comp_m1    --d_static 1152 --d_dyn 128 --lambda_comp 1.0 --comp_margin 1.0
run U2_comp_m2    --d_static 1152 --d_dyn 128 --lambda_comp 1.0 --comp_margin 2.0
run U3_comp_m05   --d_static 1152 --d_dyn 128 --lambda_comp 1.0 --comp_margin 0.5
run U4_comp_m1_w3 --d_static 1152 --d_dyn 128 --lambda_comp 3.0 --comp_margin 1.0
#  at TODAY's rate/shape -- does the hinge fix the allocation without changing sizes?
run U5_comp_today --d_static 96 --static_grid 4 --d_dyn 256 --lambda_comp 1.0 --comp_margin 1.0
#  hinge AND the structural constraint together
run U6_comp_bd    --d_static 1152 --d_dyn 128 --lambda_comp 1.0 --comp_margin 1.0 --decoder basedelta
#  control: same rate, no hinge
run U7_nohinge    --d_static 1152 --d_dyn 128
echo "PHASEU_DONE"
