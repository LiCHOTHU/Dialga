#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=01:30:00 --mem-per-gpu=100G --job-name=swap_probe
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Latent-swap (controllable-editing / semantics demo): identity_follows_A + motion_follows_B,
# on the +indep model vs the no-loss baseline. Does the independence loss make the swap clean?
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); case "$HN" in *"$BAD"*) echo guard; sleep 60; sbatch "$WD/scripts/sbatch/swap_probe.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
for tag in clevrer_distill clevrer_disent; do
  CK=$(ls $CED/$tag/ckpt_ep0200.pt $CED/$tag/v5_best.pt 2>/dev/null | head -1)
  echo "==================== $tag ($CK) ===================="
  python -u scripts/probes/latent_swap.py --ckpt "$CK" --cache_dir "$WAN" --n_pairs 300 \
    --out "$OUT/swap_${tag}.json" 2>&1 | grep -iE "identity_follows|motion_follows|no.swap|control|verdict"
done
echo SWAP_PROBE_DONE
