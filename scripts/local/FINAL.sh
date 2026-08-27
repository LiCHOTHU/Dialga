#!/usr/bin/env bash
# FINAL COMMIT BLOCK. Target: VideoFlexTok's compression (1152 floats = 24.0x).
# Candidate: base+delta, z_static 576 (9ch 8x8), z_dyn 64/frame (1ch 8x8).
#   measured single-seed: 32.35 dB, swap +0.053, zs_cost 524%, zd_cost 150%, CKA 0.325
# Three seeds, plus the two arms that decide the remaining open questions:
#   - decomposed teacher: the ONLY intervention that lowered code overlap (0.347->0.327)
#   - ENTANGLED control:  it passed the swap test at 7680 floats, so the swap number
#                         needs its control at THIS rate too before anything is claimed
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/FINAL; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 60 \
   --batch_size 16 --epochs 60 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0 \
   --static_grid 8 --d_static 576 --d_dyn 64 --decoder basedelta"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
for S in 0 1 2; do run F_cand_s$S --seed $S; done
run F_teach_s0  --seed 0 --dino_cache_dir outputs/cache/dino_clevrer_W33 \
                --lambda_dino 0.5 --lambda_dyn_teach 0.5
run F_entang_s0 --seed 0 --shared_trunk
run F_grid_s0   --seed 0 --decoder grid
echo "FINAL_DONE"
