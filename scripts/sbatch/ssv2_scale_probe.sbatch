#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=03:00:00 --mem-per-gpu=100G --job-name=ssv2_scale_probe
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# DATA-SCALING result: probe the 8k-trained and 32k-trained SSv2 models on the
# identical val protocol -> does 4x data lift SSv2 top-1? Self-chains off bad node.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*) echo "[guard] bad node -> sleep+resubmit"; sleep 60; sbatch "$WD/scripts/sbatch/ssv2_scale_probe.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
TR=/storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33
VA=/storage/scratch1/8/lwang831/cache/ssv2_val2000_W33
for tag in ssv2_train ssv2_train_32k; do
  CK=$(ls $CED/$tag/ckpt_ep0200.pt $CED/$tag/v5_best.pt $CED/$tag/last.pt 2>/dev/null | head -1)
  echo "==================== MODEL: $tag  ($CK) ===================="
  [ -z "$CK" ] && { echo "  no checkpoint for $tag"; continue; }
  python -u scripts/probes/ssv2_action_probe.py --ckpt "$CK" --train_cache "$TR" --val_cache "$VA" \
    2>&1 | grep -E "feature|z_dyn|z_static|raw Wan|both|chance|dim.*top1|----"
done
echo SSV2_SCALE_PROBE_DONE
