#!/bin/bash
# Cluster-adapted wrapper to build the 10k-vid Wan-VAE cache on this H200.
# Resume-safe: cache_wan_latents.py skips existing latents/<idx>.pt files.
set -u
cd /storage/home/hcoda1/8/lwang831/workspace/Dialga

RIVER_PY=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
export TMPDIR=/storage/project/r-agarg35-0/lwang831/tmp
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE=/storage/project/r-agarg35-0/lwang831/hf_cache/hub

OUT_DIR=/storage/scratch1/8/lwang831/cache/wan_10000vid_W12
LOG_DIR=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/logs
mkdir -p "$LOG_DIR" "$OUT_DIR/latents"
LOG="$LOG_DIR/cache_10000.log"
TARGET=40000
MAX_RESTARTS=20

n=0
while [ $n -lt $MAX_RESTARTS ]; do
  have=$(ls "$OUT_DIR/latents" 2>/dev/null | wc -l)
  if [ "$have" -ge "$TARGET" ]; then
    echo "[wrapper] $have/$TARGET — TARGET reached, exiting." | tee -a "$LOG"
    break
  fi
  n=$((n+1))
  echo "[wrapper] attempt $n  have=$have/$TARGET  $(date)" | tee -a "$LOG"
  $RIVER_PY -u scripts/cache_wan_latents.py \
    --data_dir /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
    --annotation_dir /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations \
    --max_videos 10000 --windows_per_video 4 --window_length 12 \
    --frames_per_video 128 --image_size 128 \
    --out_dir "$OUT_DIR" --device cuda --dtype float16 >> "$LOG" 2>&1
  ec=$?
  echo "[wrapper] python exited with code $ec  $(date)" | tee -a "$LOG"
  [ $ec -eq 0 ] && break
  sleep 5
done
final=$(ls "$OUT_DIR/latents" 2>/dev/null | wc -l)
echo "[wrapper] DONE  final_count=$final  $(date)" | tee -a "$LOG"
touch "$OUT_DIR/CACHE_DONE.marker"
