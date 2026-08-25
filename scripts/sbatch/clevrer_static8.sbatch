#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=08:00:00 --mem-per-gpu=80G --job-name=clevrer_static8
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# RECON FIX (structure-preserving): make z_static FULL-RESOLUTION spatial (static_grid
# 4 -> 8, matching z_dyn and the 8x8 Wan latent) so appearance detail is not
# downsampled 2x before the decoder. Same factorization (static/dyn split + indep +
# VICReg + InfoNCE) -- ONLY the static spatial resolution changes. Compare recon +
# leakage vs clevrer_disent (static_grid=4).
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"; PT=/storage/project/r-agarg35-0/lwang831/tmp
HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *atl1-1-03-007-29-0*|*atl1-1-03-007-31-0*) echo "[guard] bad node -> sleep+resubmit"; sleep 90; sbatch --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0 "$WD/scripts/sbatch/clevrer_static8.sbatch"; exit 0;; esac
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33; DINO=/storage/scratch1/8/lwang831/cache/dino_patch_10000vid_W33
source ~/.bashrc; conda activate river
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} TMPDIR="$PT" MPLCONFIGDIR="$PT/matplotlib"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1
OUT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/clevrer_static8; mkdir -p "$OUT"
SP="$WD/scripts/sbatch/clevrer_static8.sbatch"; A="${ATTEMPT:-1}"
[ -f "$OUT/DONE" ] && { echo DONE; exit 0; }
[ "$A" -lt 500 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((A+1))" --exclude=atl1-1-03-007-29-0,atl1-1-03-007-31-0 "$SP" || true
# ONLY change vs clevrer_disent: static_grid 4->8, d_static 96->256 (4 ch/pos at 8x8).
python -u scripts/train_v5.py --dataset clevrer --cache_dir "$WAN" --dino_cache_dir "$DINO" --out_dir "$OUT" \
  --max_videos 0 --batch_size 8 --num_workers 6 --epochs 200 --stage1_epochs 5 --stage2_epochs 195 \
  --lr 3e-4 --lr_schedule constant --weight_decay 1e-3 \
  --d_static 256 --d_dyn 256 --d_state 48 --enc_hidden_ch 192 --dec_hidden_ch 384 \
  --static_grid 8 --pool_type spatial --decoder_type spatial --dyn_spatial --dyn_grid 8 \
  --lambda_recon 1.0 --lambda_pred 0.1 --lambda_fwd 0.1 --lambda_consist 1.0 \
  --lambda_mae 0.5 --lambda_attrs 0 --lambda_event_aux 0.1 --lambda_gate 0.1 --lambda_pixel 0 \
  --lambda_indep 1.0 --lambda_vic_var 0.05 --lambda_vic_cov 0.05 \
  --infonce_temperature 0.1 --early_stop_patience 0 --val_frac 0.1 --val_every 5 \
  --log_every 20 --ckpt_every 10 --resume auto --vae_dtype bfloat16 \
  --wandb_project dialga --wandb_run_name clevrer_static8 --device cuda
echo "Done $(date)"
