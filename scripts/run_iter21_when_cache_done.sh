#!/bin/bash
# Wait for cache_10000 to finish, then launch Iter 21 training.
set -u
cd /home/licho/workspace/Dialga
source ~/anaconda3/etc/profile.d/conda.sh
conda activate river

CACHE=outputs/cache/wan_10000vid_W12
TLOG=outputs/logs/train_iter21.log
TARGET=40000

echo "[launcher] waiting for cache to reach $TARGET windows…  $(date)" | tee -a "$TLOG"
while true; do
  have=$(ls "$CACHE/latents" 2>/dev/null | wc -l)
  meta_ok=$([ -f "$CACHE/metadata.json" ] && echo yes || echo no)
  if [ "$have" -ge "$TARGET" ] && [ "$meta_ok" = yes ]; then
    echo "[launcher] cache complete: have=$have, metadata=$meta_ok  $(date)" | tee -a "$TLOG"
    break
  fi
  sleep 30
done

OUT=outputs/iter21_10000vid_$(date +%H%M%S)
echo "[launcher] launching training to $OUT" | tee -a "$TLOG"
python -u scripts/train_trajectory.py \
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
