#!/usr/bin/env bash
# Stage 3. Fills the gaps stage 2 left: CKA (it crashed on the CLEVRER-only loader),
# the SSv2 ablation/swap numbers for the committed seeds and the entangled control, and
# seed error bars for the CLEVRER matched-protocol table -- whose single-seed status is
# the caveat currently written into the paper.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L=outputs/logs
step () { echo "=== $1 @ $(date -Is) ==="; shift; "$@" || echo "!!! step failed rc=$?"; }

CK="outputs/FINAL_SSV2/ckpt.pt outputs/FINAL_SSV2_s1/ckpt.pt outputs/FINAL_SSV2_s2/ckpt.pt outputs/ENT_SSV2/ckpt.pt"
LB="ours ours_s1 ours_s2 entangled"

step "cka-ssv2" python -u scripts/local/overlap_eval.py --ckpts $CK --labels $LB \
  --cache_dir outputs/cache/ssv2_W17_full --out $L/ssv2_overlap.json

step "ablate-ssv2" python -u scripts/local/swap_eval.py --ckpts $CK --labels $LB \
  --cache_dir outputs/cache/ssv2_W17_full --out $L/ssv2_ablate.json

# CLEVRER matched-protocol, 3 seeds -- removes the single-seed caveat on tab:matched
for s in 0 1 2; do
  step "clevrer-matched-seed$s" python -u scripts/probes/clevrer_decode_baselines.py \
    --ckpt outputs/FINAL/ckpt.pt --seed $s --pixel \
    --methods ours videoflextok videomae dinov2 wanflat \
    --out $L/clevrer_matched_s$s.json
done

echo "QUEUE3_OK @ $(date -Is)"
