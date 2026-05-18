#!/bin/bash
# Launch Iter 21 training after the cache is complete.
# Polls for the CACHE_DONE.marker; once present, starts training.
set -u
cd /storage/home/hcoda1/8/lwang831/workspace/Dialga

RIVER_PY=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE=/storage/project/r-agarg35-0/lwang831/hf_cache/hub

CACHE=/storage/scratch1/8/lwang831/cache/wan_10000vid_W12
LOG_DIR=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/logs
mkdir -p "$LOG_DIR"
TLOG="$LOG_DIR/train_iter21.log"
TARGET=40000

echo "[launcher] waiting for cache (CACHE_DONE.marker)…  $(date)" | tee -a "$TLOG"
until [ -f "$CACHE/CACHE_DONE.marker" ]; do
  have=$(ls "$CACHE/latents" 2>/dev/null | wc -l)
  echo "  cache progress: $have/$TARGET  $(date)" | tee -a "$TLOG"
  sleep 60
done
echo "[launcher] cache complete: $(ls $CACHE/latents | wc -l) windows  $(date)" | tee -a "$TLOG"

OUT=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/iter21_10000vid_$(date +%H%M%S)
mkdir -p "$OUT"
echo "[launcher] launching training → $OUT" | tee -a "$TLOG"

$RIVER_PY -u scripts/train_trajectory.py \
  --cache_dir "$CACHE" \
  --out_dir "$OUT" \
  --epochs 60 --batch_size 4 --num_workers 0 \
  --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
  --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
  --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
  --lambda_event_sup 0.02 \
  --val_frac 0.2 --val_every 5 \
  --log_every 50 --ckpt_every 5 \
  --device cuda >> "$TLOG" 2>&1
ec=$?
echo "[launcher] training exited with code $ec  $(date)" | tee -a "$TLOG"
echo "$OUT" > /storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/iter21_latest_out.txt
touch "$OUT/TRAIN_DONE.marker"
