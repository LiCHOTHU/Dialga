#!/usr/bin/env bash
set -uo pipefail
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
cd /home/licho/workspace/Dialga; export PYTHONPATH=/home/licho/workspace/Dialga
exec python -u scripts/local/prep_ssv2.py --max_videos 6000
