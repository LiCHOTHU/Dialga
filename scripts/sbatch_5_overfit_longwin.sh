#!/bin/bash
#SBATCH --job-name=dlg_5_lw
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_lw_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_lw_%j.err

# Train/inference mismatch fix: window_length 6 → 32.
# Conv1d head (kernel=5, depth=2, RF=9) saw ALL frames as boundary-padded
# at T=6, but only 4/128 frames at inference T=128. Head learned to fire
# only in patterns that include zero-padding context — patterns that
# never arise at inference. Bumping window to 32 makes most frames see
# normal context, matching the inference regime.
#
# batch_size 8 → 4 to keep total per-step token count comparable.
# windows_per_video 8 → 4 (4×32 = 128 = full video coverage).

set -euo pipefail
REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga
mkdir -p "${OUTBASE}/slurm"
TEACHER=${TEACHER:-gt}        # gt | motion
RUN_NAME=${RUN_NAME:-slot_5_longwin_${TEACHER}_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"
cd "${REPO}"
echo "Run dir: ${OUTDIR}  TEACHER=${TEACHER}"
export PYTHONUNBUFFERED=1

EXTRA_ARGS=""
if [ "${TEACHER}" = "motion" ]; then
  EXTRA_ARGS="training.motion_abs_thresh=0.05 training.event_teacher_hard=true training.motion_teacher_gt_q=true training.self_event_sharpness=8.0 training.pixel_event_attention_sigma=0.20"
fi

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 \
    training.window_length=32 \
    training.windows_per_video=4 \
    training.batch_size=4 training.num_workers=2 \
    training.stage1_epochs=80 training.stage2_epochs=0 \
    training.ckpt_every=80 training.noise_sigma=0.0 \
    training.lambda_state=1.0 training.lambda_state_anneal_to=1.0 \
    training.lambda_state_anneal_epochs=0 training.lambda_solver=1.0 \
    training.lambda_static=0.1 training.lambda_collision=0.0 \
    training.lambda_event=1.0 training.event_pos_weight=50.0 \
    training.event_label_dilation=3 training.event_supervision=${TEACHER} \
    training.lambda_recon_stage1=1.0 training.lambda_contrastive=0.5 \
    training.log_interval=10 model.event_input_mode=qva wandb.enabled=false \
    ${EXTRA_ARGS}

"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" --max_videos 5 --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -20
