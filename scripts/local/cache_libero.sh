#!/usr/bin/env bash
# LIBERO-90 -> Wan latents. single_view_video avoids de-tiling the multi-view files.
# W=17 -> T_lat=5 matches the SSv2 config so the committed shape transfers unchanged.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
python -u scripts/local/prep_libero.py
python -u scripts/cache_wan_ssv2.py \
  --video_dir datasets/libero/videos \
  --label_json datasets/libero/labels.json \
  --split_json datasets/libero/split.json \
  --max_videos 5000 --windows_per_video 4 --window_frames 17 \
  --image_size 128 --out_dir outputs/cache/libero_W17
