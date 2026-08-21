#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=08:00:00 --mem-per-gpu=80G --job-name=ssv2_distillL
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"; PT=/storage/project/r-agarg35-0/lwang831/tmp
WAN=/storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33; DINO=/storage/scratch1/8/lwang831/cache/dino_ssv2_8000vid_W33
source ~/.bashrc; conda activate river
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TMPDIR="$PT" MPLCONFIGDIR="$PT/matplotlib"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1
OUT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/ssv2_distill_large; mkdir -p "$OUT"
SP="$WD/scripts/sbatch/ssv2_distill_large.sbatch"; A="${ATTEMPT:-1}"
[ -f "$OUT/DONE" ] && { echo DONE; exit 0; }
[ "$A" -lt 500 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((A+1))" "$SP" || true
python -u scripts/train_v5.py --dataset ssv2 --cache_dir "$WAN" --dino_cache_dir "$DINO" --out_dir "$OUT" \
  --max_videos 0 --batch_size 8 --num_workers 6 --epochs 200 --stage1_epochs 5 --stage2_epochs 195 \
  --lr 3e-4 --lr_schedule constant --weight_decay 1e-3 \
  --d_static 96 --d_dyn 256 --d_state 48 --enc_hidden_ch 192 --dec_hidden_ch 384 \
  --static_grid 4 --pool_type spatial --decoder_type spatial --dyn_spatial --dyn_grid 8 \
  --lambda_recon 1.0 --lambda_pred 0.1 --lambda_fwd 0.1 --lambda_consist 1.0 \
  --lambda_mae 0.5 --lambda_attrs 0 --lambda_event_aux 0 --lambda_gate 0 --lambda_pixel 0 \
  --infonce_temperature 0.1 --early_stop_patience 0 --val_frac 0.1 --val_every 5 \
  --log_every 20 --ckpt_every 10 --resume auto --vae_dtype bfloat16 \
  --wandb_project dialga --wandb_run_name ssv2_distill_large --device cuda
echo "Done $(date)"
