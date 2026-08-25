#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=08:00:00 --mem-per-gpu=120G --job-name=decode_pixel
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Matched-dimension PIXEL reconstruction: commit each encoder to a limited embedding
# (PCA to 96 / 384 floats), decode to pixels with the SAME head, report PSNR.
# The FAIR reconstruction fight (not vs codecs): among learned encoders at matched
# rate, is our decodable code better than the pretrained (semantic) ones?
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
# guard against BOTH known bad-ECC nodes (-29-0 and -31-0)
HN=$(hostname); case "$HN" in *atl1-1-03-007-29-0*|*atl1-1-03-007-31-0*) echo "bad node $HN"; sleep 120; sbatch --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0 "$WD/scripts/sbatch/decode_pixel.sbatch"; exit 0;; esac
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
CK=$(ls $CED/clevrer_distill/ckpt_ep0200.pt $CED/clevrer_distill/v5_best.pt 2>/dev/null | head -1)
RES="$OUT/decode_pixel_matched.json"
# self-chain on embers preemption: queue a successor (afterany) that will resume via
# the probe's skip-done logic. Stop chaining once all 5 methods are in the json.
SP="$WD/scripts/sbatch/decode_pixel.sbatch"; A="${ATTEMPT:-1}"
NDONE=$(python -c "import json,sys;print(len(json.load(open('$RES'))))" 2>/dev/null || echo 0)
[ "$NDONE" -ge 5 ] && { echo "all methods done ($NDONE)"; exit 0; }
[ "$A" -lt 20 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((A+1))" "$SP" || true
echo "==================== decode-pixel matched-dim ($CK) attempt $A, $NDONE done ===================="
# methods: ours + real pretrained encoders + full-latent ceiling (NO mean-pool).
# ours/wanflat first (no RGB, fast -> banked early); RGB encoders after.
python -u scripts/probes/clevrer_decode_baselines.py --ckpt "$CK" \
  --cache_dir "$WAN" --video_root "$VID" \
  --methods ours wanflat videomae videoflextok dinov2 \
  --max_videos 1200 --epochs 60 --dec_hidden 384 \
  --pca_to 96 384 --pixel --pixel_n 32 \
  --out "$RES" 2>&1 | grep -E "@dim|full_dim|skip|resume|DECODE|rror"
echo DECODE_PIXEL_FULL_DONE
