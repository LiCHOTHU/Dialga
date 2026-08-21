#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=08:00:00 --mem-per-gpu=80G --job-name=ssv2_scale
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
CACHE=/storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33; PT=/storage/project/r-agarg35-0/lwang831/tmp
source ~/.bashrc; conda activate river
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TMPDIR="$PT" MPLCONFIGDIR="$PT/matplotlib"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1
SIZE="${SIZE:?set SIZE=med|large}"
case $SIZE in
  med)   DS=192; DD=512; ENCH=384; DECH=512 ;;
  large) DS=384; DD=1024; ENCH=512; DECH=768 ;;
esac
OUT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/ssv2_scale_$SIZE; mkdir -p "$OUT"
SP="$WD/scripts/sbatch/ssv2_scale.sbatch"; A="${ATTEMPT:-1}"
[ -f "$OUT/DONE" ] && { echo DONE; exit 0; }
[ "$A" -lt 500 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,SIZE=$SIZE,ATTEMPT=$((A+1))" "$SP" || true
python -u scripts/train_v5.py --dataset ssv2 --cache_dir "$CACHE" --out_dir "$OUT" \
  --max_videos 0 --batch_size 8 --num_workers 6 --epochs 200 --stage1_epochs 5 --stage2_epochs 195 \
  --lr 3e-4 --lr_schedule constant --weight_decay 1e-3 \
  --d_static $DS --d_dyn $DD --d_state 48 --enc_hidden_ch $ENCH --dec_hidden_ch $DECH \
  --static_grid 4 --pool_type spatial --decoder_type spatial --dyn_spatial --dyn_grid 8 \
  --lambda_recon 1.0 --lambda_pred 0.1 --lambda_fwd 0.1 --lambda_consist 1.0 \
  --lambda_attrs 0.0 --lambda_event_aux 0.0 --lambda_gate 0.0 --lambda_pixel 0.0 --lambda_mae 0.0 \
  --infonce_temperature 0.1 --early_stop_patience 0 --val_frac 0.1 --val_every 5 \
  --log_every 20 --ckpt_every 10 --resume auto --vae_dtype bfloat16 \
  --wandb_project dialga --wandb_run_name ssv2_scale_$SIZE --device cuda
echo "Done $(date)"
