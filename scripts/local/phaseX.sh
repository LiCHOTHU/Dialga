#!/usr/bin/env bash
# PHASE X: 4x MORE compression than today (2400 -> ~576-800 floats), i.e. 35-48x,
# which is at or beyond VideoMAE's 36.0x and VideoFlexTok's 24.0x.
#
# Predicted by the PCA rate curves (0.915*static_err + 0.085*dyn_err):
#     today   (  96,256) 2400f 11.5x  err 0.314
#     X2      ( 512, 32)  800f 34.6x  err 0.100
#     X3      ( 288, 32)  576f 48.0x  err 0.161
# i.e. a QUARTER of today's bits should still beat today's error, because today's
# allocation (4% static / 96% dynamic) is far from where the information is
# (91.5% static / 8.5% dynamic). Each arm uses base+delta so the factorization is
# kept; grid twins say how much the constraint costs at these rates.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/highcompr; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
#            1152f = 24.0x  (VideoFlexTok's rate)
run X1_24x_bd  --decoder basedelta --d_static 576 --static_grid 8 --d_dyn 64
#             800f = 34.6x  (about VideoMAE's rate)
run X2_35x_bd  --decoder basedelta --d_static 512 --static_grid 8 --d_dyn 32 --dyn_grid 4
run X2_35x_gr  --d_static 512 --static_grid 8 --d_dyn 32 --dyn_grid 4
#             576f = 48.0x  (4x today's compression)
run X3_48x_bd  --decoder basedelta --d_static 288 --static_grid 4 --d_dyn 32 --dyn_grid 4
run X3_48x_gr  --d_static 288 --static_grid 4 --d_dyn 32 --dyn_grid 4
#             656f = 42.1x
run X4_42x_bd  --decoder basedelta --d_static 512 --static_grid 8 --d_dyn 16 --dyn_grid 4
echo "PHASEX_DONE"
