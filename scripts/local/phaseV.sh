#!/usr/bin/env bash
# PHASE V: base+delta is the only family that is genuinely DISENTANGLED (swap margin
# +0.049, id(swap)==id(recon)), both codes necessary (703%/273%) and reconstructs
# better than today (0.0079 vs 0.0089). Two things left:
#   1. balance it. 703/273 is 0.39; a GENTLE hinge should even it without the collapse
#      U6 hit (lambda_comp 1.0 + margin 1.0 on base+delta -> recon 1.0269, a 100x blowup,
#      because base+delta MAKES z_static alone good while the hinge demands it be bad).
#   2. cheapen it. It runs at 7680 floats (3.6x) vs today's 2400 (11.5x); does the
#      disentanglement survive at lower rate?
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/comp; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 30 \
   --batch_size 16 --epochs 60 --seed 0 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0 --static_grid 8 --decoder basedelta"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
# gentle hinge on base+delta: balance without collapse
run V1_bd_h03_m03 --d_static 768 --d_dyn 768 --lambda_comp 0.3 --comp_margin 0.3
run V2_bd_h01_m05 --d_static 768 --d_dyn 768 --lambda_comp 0.1 --comp_margin 0.5
# cheaper base+delta: does disentanglement survive at today's-ish rate?
run V3_bd_1152_128 --d_static 1152 --d_dyn 128            # 2304 floats, 12.0x
run V4_bd_768_192  --d_static 768  --d_dyn 192            # 2496 floats, 11.1x
run V5_bd_576_64   --d_static 576  --d_dyn 64             # 1152 floats, 24.0x
echo "PHASEV_DONE"
