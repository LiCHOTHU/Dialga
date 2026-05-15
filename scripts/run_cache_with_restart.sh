#!/bin/bash
# Wrapper that auto-restarts the cache job until all 40000 windows are on disk.
# Combined with the resume feature in cache_wan_latents.py, this is robust to
# silent deaths from harness/session lifecycle issues.
set -u
cd /home/licho/workspace/Dialga
source ~/anaconda3/etc/profile.d/conda.sh
conda activate river

OUT_DIR=outputs/cache/wan_10000vid_W12
LOG=outputs/logs/cache_10000.log
TARGET=40000
MAX_RESTARTS=50

n=0
while [ $n -lt $MAX_RESTARTS ]; do
  have=$(ls "$OUT_DIR/latents" 2>/dev/null | wc -l)
  if [ "$have" -ge "$TARGET" ]; then
    echo "[wrapper] $have/$TARGET — TARGET reached, exiting wrapper." | tee -a "$LOG"
    break
  fi
  n=$((n+1))
  echo "[wrapper] attempt $n  have=$have/$TARGET  $(date)" | tee -a "$LOG"
  python -u scripts/cache_wan_latents.py \
    --max_videos 10000 --windows_per_video 4 --window_length 12 \
    --frames_per_video 128 --image_size 128 \
    --out_dir "$OUT_DIR" --device cuda --dtype float16 >> "$LOG" 2>&1
  ec=$?
  echo "[wrapper] python exited with code $ec  $(date)" | tee -a "$LOG"
  sleep 5
done
echo "[wrapper] DONE  final_count=$(ls $OUT_DIR/latents | wc -l)" | tee -a "$LOG"
