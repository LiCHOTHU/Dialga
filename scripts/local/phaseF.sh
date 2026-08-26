#!/usr/bin/env bash
# PHASE F: DINOv2 as the SEMANTIC teacher -- the one lever left after B/C showed that
# semantics is flat (attr mAP 0.72-0.75) under every rate, shape, memory and
# median-teacher variant, and that InfoNCE is the only term that moves it at all.
#
# F1 vs F2 is the test that has been outstanding all session: train_v5's
# AuxSemanticDecoder feeds the DINO target BOTH z_static and z_dyn, so the semantic
# signal can be satisfied through z_dyn -- which has 24x the rate and is the cheaper
# path -- while lambda_indep is simultaneously pushing identity OUT of z_dyn. Those two
# terms fight. F1 routes DINO to z_static ALONE; F2 is the current behaviour.
#
# Caveat to carry into any write-up: distilling from DINOv2 means z_static's semantics
# come FROM DINOv2, so the model can never honestly be compared against DINOv2 as a
# baseline. The paper already flags this ("it is our own MAE teacher").
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/overnight; mkdir -p "$OUT"
D=outputs/cache/dino_clevrer_W33
BASE="--cache_dir outputs/cache/clevrer_W33_10k --epochs 30 --preload --max_videos 2000 \
      --eval_every 10 --seed 0 --batch_size 16 --d_static 96 --static_grid 4 \
      --dino_cache_dir $D"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $BASE --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
run F1_dino_static    --lambda_dino 0.5 --dino_to static
run F2_dino_both      --lambda_dino 0.5 --dino_to both
run F3_dino_static_2  --lambda_dino 2.0 --dino_to static
run F4_dino_i0        --lambda_dino 0.5 --dino_to static --lambda_indep 0.0
run F5_dino_tgt       --lambda_dino 0.5 --dino_to static \
                      --static_target video_median --lambda_static_tgt 1.0
run F6_dino_static_01 --lambda_dino 0.1 --dino_to static
echo "PHASEF_DONE"
