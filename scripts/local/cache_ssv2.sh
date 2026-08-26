#!/usr/bin/env bash
set -uo pipefail
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
cd /home/licho/workspace/Dialga; export PYTHONPATH=/home/licho/workspace/Dialga
# W=17 (=4*4+1 -> T_lat=5), 4 ordered windows per clip. SSv2 is median 45 frames, so
# W=33 would force ~88% window overlap and there would be nothing for a cross-chunk
# memory to accumulate; W=17 keeps 98.7% of clips usable with ~45% overlap.
exec python -u scripts/cache_wan_ssv2.py \
  --video_dir datasets/ssv2/videos \
  --label_json datasets/ssv2/labels.json \
  --split_json datasets/ssv2/split.json \
  --max_videos 6000 --windows_per_video 4 --window_frames 17 \
  --image_size 128 --out_dir outputs/cache/ssv2_W17_6k
