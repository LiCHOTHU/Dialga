#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=03:00:00 --mem-per-gpu=180G --job-name=ssv2_mlp
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# THE ceiling test: linear vs nonlinear (MLP) probe on the FULL Wan latent (wanflat,
# 27648-d) vs our compressed code, on SSv2. Answers "are we failing because the info
# isn't in the latent (input-bound) OR because it's there but we don't extract it
# (fixable)". If wanflat-MLP >> ours -> we are NOT hitting the upper bound.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
HN=$(hostname); case "$HN" in *atl1-1-03-007-29-0*|*atl1-1-03-007-31-0*) echo "bad node $HN"; sleep 120; sbatch --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0 "$WD/scripts/sbatch/ssv2_mlp_probe.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
CK=$(ls $CED/ssv2_disent/v5_best.pt $CED/ssv2_disent/ckpt_ep0200.pt 2>/dev/null | head -1)
echo "==================== SSv2 linear-vs-MLP ceiling test ($CK) ===================="
python -u scripts/probes/ssv2_action_probe.py --ckpt "$CK" \
  --train_cache /storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33 \
  --val_cache   /storage/scratch1/8/lwang831/cache/ssv2_val2000_W33 \
  --mlp 2>&1 | grep -vE "pymp|shutil|onerror|onexc|rmtree|dir_fd|Device or resource|FutureWarning|warnings.warn"
echo SSV2_MLP_FULL_DONE
