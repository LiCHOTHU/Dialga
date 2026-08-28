#!/usr/bin/env bash
# Fill tab:matched (Table 3) with the retrained committed-config CLEVRER models, 3 seeds.
# Rate: 576 + 64*9 = 1152 floats = 24.0x on the (48,9,8,8) CLEVRER latent.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
for s in 0 1 2; do
  echo "=== clevrer-matched seed $s @ $(date -Is) ==="
  python -u scripts/probes/clevrer_decode_baselines.py \
    --ckpt outputs/FINAL_CLEVRER_s$s/ckpt.pt \
    --cache_dir outputs/cache/clevrer_W33_10k \
    --video_root datasets/CLEVRER/train_video \
    --methods ours videoflextok videomae dinov2 wanflat \
    --seed $s --pixel \
    --out outputs/logs/clevrer_matched_s$s.json || echo "!!! seed $s failed rc=$?"
done
echo "QUEUE6_OK @ $(date -Is)"
