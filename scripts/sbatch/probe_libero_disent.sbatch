#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=02:00:00 --mem-per-gpu=80G --job-name=probe_libero_disent
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Answers: with the independence loss, which code (z_static vs z_dyn) predicts ACTION
# best, and is it better than the no-loss baseline? Waits for libero_disent, self-resubmits.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
CACHE=/storage/scratch1/8/lwang831/cache/libero_90_wan_W33
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
A="${ATTEMPT:-1}"
if [ ! -f "$CED/libero_disent/DONE" ] && [ ! -f "$CED/libero_disent/ckpt_ep0200.pt" ]; then
  echo "[wait] libero_disent not ready ($A)"; [ "$A" -lt 40 ] && { sleep 1200; sbatch --export="ALL,ATTEMPT=$((A+1))" "$WD/scripts/sbatch/probe_libero_disent.sbatch"; }; exit 0
fi
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1
for tag in libero_disent libero_v59; do
  CK=$(ls $CED/$tag/ckpt_ep0200.pt $CED/$tag/v5_best.pt $CED/$tag/last.pt 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "== $tag: no ckpt =="; continue; }
  echo "==================== $tag ($CK) ===================="
  python -u scripts/probes/libero_action_probe.py --ckpt "$CK" --cache_dir "$CACHE" \
    --out "$OUT/action_${tag}.json" 2>&1 | grep -iE "z_dyn|z_static|wanmean|random|action|MSE|acc" | tail -8
done
echo PROBE_LIBERO_DONE
