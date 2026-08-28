#!/usr/bin/env bash
# Seed error bars for the SSv2 reconstruction table (Q3b), which is single-seed as
# written. Features are cached on the first pass, so seeds 1 and 2 skip the webm
# decoding that dominates the runtime.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
for s in 0 1 2; do
  echo "=== ssv2-recon seed $s @ $(date -Is) ==="
  python -u scripts/local/ssv2_decode_baselines.py \
    --methods ours wanflat wanmean videomae dinov2 videoflextok \
    --n_train 4000 --n_val 600 --n_pixel 96 --epochs 60 --seed $s \
    --feat_cache outputs/cache/ssv2_decode_feats \
    --out outputs/logs/ssv2_decode_seed$s.json || echo "!!! seed $s failed rc=$?"
done
echo "QUEUE4_OK @ $(date -Is)"
