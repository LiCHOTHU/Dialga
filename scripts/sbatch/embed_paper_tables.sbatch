#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=180G
#SBATCH --job-name=embed_tables
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# The remaining publication tables for the EMBEDDING paper. All eval-only (no training),
# so all of it lands within the one-week deadline. See EXPERIMENT_PLAN_v7.md.
#
# BLOCK A — THE ATTRIBUTE-SUPERVISION CONTROL. **This decides whether the paper survives.**
#   Our headline is "a frozen linear probe recovers object attributes from z_static"
#   (colour mAP 0.932 vs 0.498 prior, beating the raw 27648-float latent). But the encoder
#   was TRAINED with a supervised AttrsHead at lambda_attrs=0.5 backpropping into it. The
#   first thing a reviewer will say is: you supervised the attributes in, then probed them
#   back out — that is circular.
#   v512_clevrer_noattrs (lambda_attrs=0.0) is the control. All four v512 ablations are
#   single-variable vs v512_clevrer_big and ALL SHARE ep70, so this is fully epoch-matched:
#       noattrs  lambda_attrs=0.0   <- the control
#       noevent  lambda_attrs=0.5   (event_aux off)
#       nopixel  lambda_attrs=0.5   (pixel off)
#       nopred   lambda_attrs=0.5   (pred off)
#   v512_clevrer_big cannot join — it only has ep130..200, no overlap. Not needed: the three
#   attrs-on arms bracket the attrs-on condition and give a spread rather than one point.
#   READ: if noattrs collapses toward the prior -> our claim is "we supervised it", and the
#   paper's headline must be rewritten. If noattrs holds up -> the claim is self-supervised
#   and much stronger.
#
# BLOCK B — does the LABEL-FREE arm still beat the baselines a reviewer names?
#   Full baseline table (PCA-96 dim-matched / random-init / wan_flat 27648 / DINOv2) on
#   noattrs@ep70. This is Block A's finding placed against real competition.
#
# BLOCK C — E3: probe mAP vs RATE (the FlexTok App-B style rate sweep, zero new training).
#   Width-matched arms at ep50: 644 / 964 / 1924 / 3844 / 7684 floats.
#   NOTE: our sweep retrains per budget = structurally FlexTok's *TiTok* arm, not the nested
#   FlexTok arm. Report it as such; do not imply nesting.

set -eo pipefail
set -x
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
DINO_CACHE="/storage/scratch1/8/lwang831/cache/dino_patch_10000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/storage/project/r-agarg35-0/lwang831/torch_home"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
OUT="${WORKDIR}/outputs/embed_paper"; mkdir -p "${OUT}"
nvidia-smi || true
set +x

echo "######## BLOCK A: attribute-supervision control, epoch-matched @ep70 ########"
python -u scripts/probes/frozen_set_probe.py \
    --ckpts \
        noattrs_L0.0_ep70="${CEDAR}/v512_clevrer_noattrs/ckpt_ep0070.pt" \
        noevent_L0.5_ep70="${CEDAR}/v512_clevrer_noevent/ckpt_ep0070.pt" \
        nopixel_L0.5_ep70="${CEDAR}/v512_clevrer_nopixel/ckpt_ep0070.pt" \
        nopred_L0.5_ep70="${CEDAR}/v512_clevrer_nopred/ckpt_ep0070.pt" \
    --epochs 400 \
    --out "${OUT}/attrs_control_ep70.json"

echo
echo "######## BLOCK B: does the LABEL-FREE arm beat the real baselines? ########"
python -u scripts/probes/baseline_probe_table.py \
    --ckpt "${CEDAR}/v512_clevrer_noattrs/ckpt_ep0070.pt" \
    --dino_cache_dir "${DINO_CACHE}" \
    --epochs 400 --pca_dim 96 \
    --out "${OUT}/baselines_noattrs_ep70.json"

echo
echo "######## BLOCK C: E3 — probe mAP vs rate, width-matched @ep50 ########"
python -u scripts/probes/frozen_set_probe.py \
    --ckpts \
        rate0644_ep50="${CEDAR}/v514_wide_lowrate/ckpt_ep0050.pt" \
        rate0964_ep50="${CEDAR}/v55_pool_mean/ckpt_ep0050.pt" \
        rate1924_ep50="${CEDAR}/v56_latent_x2/ckpt_ep0050.pt" \
        rate3844_ep50="${CEDAR}/v56_latent_x4/ckpt_ep0050.pt" \
        rate7684_ep50="${CEDAR}/v56_latent_x8/ckpt_ep0050.pt" \
    --epochs 400 \
    --out "${OUT}/rate_curve_ep50.json"

echo "EMBED_TABLES_SBATCH_DONE"
