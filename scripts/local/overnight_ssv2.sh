#!/usr/bin/env bash
# OVERNIGHT: commit the final config, then run it on the FULL SSv2 dataset.
#
# Stages, each independently resumable (every one skips work already done, so a
# SIGKILL at any point costs only the current step):
#   1. wait for the FINAL 24x validation block
#   2. extract all ~193,690 SSv2 clips from the split tar        ~16 min
#   3. Wan-encode them, W=17 -> T_lat=5, 4 ordered windows/clip  ~9.5 h, ~48 GB
#   4. train the committed config on the full set               ~2.3 h
#
# Committed config (24.0x on CLEVRER; on SSv2 T_lat=5 so the same SHAPE is
# 576 + 5x64 = 896 floats against a 15,360-float latent = 17.1x):
#   base+delta decoder, z_static 576 (9ch 8x8), z_dyn 64/frame (1ch 8x8),
#   lambda_indep 0, lambda_consist 3, median-mosaic teacher.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga

echo "[1/4] waiting for the FINAL 24x block"
while ! grep -q FINAL_DONE outputs/logs/FINAL.log 2>/dev/null; do sleep 60; done
echo "[1/4] done @ $(date -Is)"

echo "[2/4] extracting full SSv2 @ $(date -Is)"
python -u scripts/local/prep_ssv2.py --max_videos 200000 \
    --video_out datasets/ssv2/videos --meta_out datasets/ssv2
echo "[2/4] done @ $(date -Is); clips: $(ls datasets/ssv2/videos | wc -l)"

echo "[3/4] Wan-encoding full SSv2 @ $(date -Is)"
python -u scripts/cache_wan_ssv2.py \
    --video_dir datasets/ssv2/videos \
    --label_json datasets/ssv2/labels.json \
    --split_json datasets/ssv2/split.json \
    --max_videos 200000 --windows_per_video 4 --window_frames 17 \
    --image_size 128 --out_dir outputs/cache/ssv2_W17_full
echo "[3/4] done @ $(date -Is)"

echo "[4/4] training the committed config on full SSv2 @ $(date -Is)"
python -u scripts/local/train_memory.py \
    --dataset ssv2 --chunk_size_lat 5 \
    --cache_dir outputs/cache/ssv2_W17_full \
    --out_dir outputs/FINAL_SSV2 \
    --epochs 60 --batch_size 16 --eval_every 10 --seed 0 --preload \
    --decoder basedelta --static_grid 8 --d_static 576 --d_dyn 64 \
    --lambda_indep 0 --lambda_consist 3 \
    --static_target video_median --lambda_static_tgt 1.0
echo "SSV2_FULL_DONE @ $(date -Is)"
