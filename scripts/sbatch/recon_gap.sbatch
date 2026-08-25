#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=02:00:00 --mem-per-gpu=120G --job-name=recon_gap
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# CURRENT-model reconstruction PSNR vs the Wan-VAE ceiling (the stale table shows a
# prior model at 33.8). Answers "how far are we from the 48 dB upper bound NOW".
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
HN=$(hostname); case "$HN" in *atl1-1-03-007-29-0*|*atl1-1-03-007-31-0*) echo "bad node $HN"; sleep 120; sbatch --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0 "$WD/scripts/sbatch/recon_gap.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33
VID=/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
for tag in clevrer_distill clevrer_disent; do
  CK=$(ls $CED/$tag/ckpt_ep0200.pt $CED/$tag/v5_best.pt 2>/dev/null | head -1)
  echo "==================== recon gap: $tag ($CK) ===================="
  python -u scripts/eval_pixel_metrics.py --ckpt "$CK" --cache_dir "$WAN" \
    --video_dir "$VID" --n_chunks 400 --batch_size 4 \
    --out_json "$OUT/recon_gap_${tag}.json" 2>&1 \
    | grep -viE "pymp|shutil|onerror|onexc|rmtree|dir_fd|Device or resource|Warning|warn"
done
echo RECON_GAP_DONE
