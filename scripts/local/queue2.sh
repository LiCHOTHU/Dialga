#!/usr/bin/env bash
# Stage 2: runs after the overnight queue finishes. Adds seed error bars to the SSv2
# reconstruction row (currently single-seed) and the CKA overlap comparison against the
# entangled control, which only becomes possible once that control has trained.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L=outputs/logs
step () { echo "=== $1 @ $(date -Is) ==="; shift; "$@" || echo "!!! step failed rc=$?"; }

while pgrep -f overnight_queue.sh >/dev/null; do sleep 120; done

# seed error bars on our SSv2 reconstruction row
for s in _s1 _s2; do
  step "ssv2recon-ours$s" python -u scripts/local/ssv2_decode_baselines.py \
    --ckpt outputs/FINAL_SSV2$s/ckpt.pt --methods ours \
    --n_train 4000 --n_val 600 --n_pixel 96 --epochs 60 \
    --out $L/ssv2_decode_ours$s.json
done

# code overlap: committed model vs the entangled control
if [ -f outputs/ENT_SSV2/ckpt.pt ]; then
  step "cka" python -u scripts/local/overlap_eval.py \
    --ckpts outputs/FINAL_SSV2/ckpt.pt outputs/FINAL_SSV2_s1/ckpt.pt \
            outputs/FINAL_SSV2_s2/ckpt.pt outputs/ENT_SSV2/ckpt.pt \
    --labels ours ours_s1 ours_s2 entangled \
    --cache_dir outputs/cache/ssv2_W17_full --out $L/ssv2_overlap.json
  # entangled reconstruction row, for the same table
  step "ssv2recon-entangled" python -u scripts/local/ssv2_decode_baselines.py \
    --ckpt outputs/ENT_SSV2/ckpt.pt --methods ours \
    --n_train 4000 --n_val 600 --n_pixel 96 --epochs 60 \
    --out $L/ssv2_decode_entangled.json
else
  echo "!!! no entangled ckpt; skipped CKA"
fi
echo "QUEUE2_OK @ $(date -Is)"
