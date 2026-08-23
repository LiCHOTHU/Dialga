#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=05:00:00 --mem-per-gpu=120G --job-name=probe_disent
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Auto-probe the +indep models for H1/H2/H3 once they finish training. Polls for the
# DONE markers; if the CLEVRER model isn't ready, sleeps + resubmits itself. When ready,
# runs: H1/H2 identity-leakage cross-probe, H3 semantic-efficiency (rate+label curves),
# H3 decodability -- all on the NEW disentanglement-loss model. SSv2 motion probe too if ready.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*) echo "[guard] bad node"; sleep 60; sbatch "$WD/scripts/sbatch/probe_disent.sbatch"; exit 0;; esac
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
A="${ATTEMPT:-1}"
# wait for the CLEVRER +indep model; self-resubmit (max ~40 attempts * 20min gap = ~13h)
if [ ! -f "$CED/clevrer_disent/DONE" ] && [ ! -f "$CED/clevrer_disent/ckpt_ep0200.pt" ]; then
  echo "[wait] clevrer_disent not ready (attempt $A); resubmitting in 20min"
  [ "$A" -lt 40 ] && { sleep 1200; sbatch --export="ALL,ATTEMPT=$((A+1))" "$WD/scripts/sbatch/probe_disent.sbatch"; }
  exit 0
fi
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR"
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33
CK=$(ls $CED/clevrer_disent/ckpt_ep0200.pt $CED/clevrer_disent/v5_best.pt $CED/clevrer_disent/last.pt 2>/dev/null | head -1)
echo "[H1/H2] identity-leakage cross-probe on $CK"
python -u scripts/probes/clevrer_semantic_probe.py --ckpt "$CK" --cache_dir "$WAN" --max_videos 1500 \
  --out "$OUT/disent_H2_clevrer_disent.json" 2>&1 | grep -E "\[Q1\]|\[Q2 leakage\]"
echo "[H3] semantic efficiency (rate + label curves)"
python -u scripts/probes/semantic_efficiency.py --ckpt "$CK" --cache_dir "$WAN" \
  --out "$OUT/disent_H3_semantic_efficiency.json" 2>&1 | grep -E "\[rate\]|\[label\]" | tail -20
echo "[H3] decodability (ours + wan controls)"
python -u scripts/probes/clevrer_decode_baselines.py --ckpt "$CK" --cache_dir "$WAN" \
  --video_root /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
  --methods ours random wanmean wanflat --max_videos 1500 --epochs 100 \
  --out "$OUT/disent_H3_decode.json" 2>&1 | grep -E "@dim|full_dim"
# SSv2 motion probe if that model is ready too
if [ -f "$CED/ssv2_disent/DONE" ] || [ -f "$CED/ssv2_disent/ckpt_ep0200.pt" ]; then
  SCK=$(ls $CED/ssv2_disent/ckpt_ep0200.pt $CED/ssv2_disent/v5_best.pt 2>/dev/null | head -1)
  echo "[H2 downstream] SSv2 motion probe on $SCK"
  python -u scripts/probes/ssv2_action_probe.py --ckpt "$SCK" \
    --train_cache /storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33 \
    --val_cache /storage/scratch1/8/lwang831/cache/ssv2_val2000_W33 2>&1 | grep -E "both|z_dyn " | tail -4
fi
echo PROBE_DISENT_DONE
