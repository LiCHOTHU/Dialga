#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
# NOTE: the sentinel is printed by the python script itself (CEILING_OK) only on
# success. Echoing a completion marker unconditionally after the command -- which the
# previous version did -- makes the restart wrapper treat a crash as success, which is
# exactly how this job died at 11:06 and left the GPU idle for two hours.
python -u scripts/local/ssv2_ceiling.py --cache_dir outputs/cache/ssv2_W17_full
