#!/bin/bash
#SBATCH --job-name=dlg_5_g2
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_g2_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_g2_%j.err

# GT baseline v2: same as v1 but pos_weight=50 (was 20).
# v1 trained to event loss = 0.047 but never crossed sigmoid=0.5 at inference;
# pos_weight=50 should push the model to higher-confidence predictions.

set -euo pipefail
REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga
mkdir -p "${OUTBASE}/slurm"
RUN_NAME=${RUN_NAME:-slot_5_gt_v2_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"
cd "${REPO}"
echo "Run dir: ${OUTDIR}"
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 training.windows_per_video=8 \
    training.batch_size=8 training.num_workers=2 \
    training.stage1_epochs=80 training.stage2_epochs=0 \
    training.ckpt_every=80 training.noise_sigma=0.0 \
    training.lambda_state=1.0 training.lambda_state_anneal_to=1.0 \
    training.lambda_state_anneal_epochs=0 training.lambda_solver=1.0 \
    training.lambda_static=0.1 training.lambda_collision=0.0 \
    training.lambda_event=1.0 training.event_pos_weight=50.0 \
    training.event_label_dilation=3 training.event_supervision=gt \
    training.lambda_recon_stage1=1.0 training.lambda_contrastive=0.5 \
    training.log_interval=10 model.event_input_mode=qva wandb.enabled=false

"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" --max_videos 5 --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -20
