#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=03:00:00 --mem-per-gpu=120G --job-name=collision_probe
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Physical-reasoning: collision detection (NO pooling -- full temporal z_dyn) on the
# +indep model AND the no-loss baseline, full scale. z_dyn (motion) vs z_static
# (appearance control) vs raw-latent / random baselines.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); case "$HN" in *"$BAD"*) echo guard; sleep 60; sbatch "$WD/scripts/sbatch/collision_probe.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
for tag in clevrer_disent clevrer_distill; do
  CK=$(ls $CED/$tag/ckpt_ep0200.pt $CED/$tag/v5_best.pt 2>/dev/null | head -1)
  echo "==================== $tag ($CK) ===================="
  python -u scripts/probes/collision_probe.py --ckpt "$CK" --cache_dir "$WAN" \
    --max_videos 2500 --out "$OUT/collision_${tag}.json" 2>&1 | grep -E "\[data\]|AP=|COLLISION"
done
echo COLLISION_FULL_DONE
