#!/usr/bin/env bash
# Overnight chain: finish the SSv2 PSNR table, then the LIBERO action table and the
# entangled control the control rows need. Each step is independent -- a failure logs
# and moves on rather than stranding the GPU for the rest of the night.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
L=outputs/logs; mkdir -p $L
step () { echo "=== $1 @ $(date -Is) ==="; shift; "$@" || echo "!!! step failed rc=$?"; }

# wait out the PSNR job still holding the GPU
while pgrep -f ssv2_decode_baselines.py >/dev/null; do sleep 60; done

# 1. substrate ceiling: true latent -> VAE -> pixels, no decode head
step "vae-ceiling" python -u scripts/local/ssv2_decode_baselines.py \
  --methods vae --n_train 200 --n_val 600 --n_pixel 96 --epochs 1 \
  --out $L/ssv2_decode_vae.json

# 2. LIBERO action table: committed model (3 seeds) + untrained control
for s in "" _s1 _s2; do
  step "libero-ours$s" python -u scripts/local/libero_probe.py \
    --ckpt outputs/FINAL_SSV2$s/ckpt.pt --label ours$s --out $L/libero_ours$s.json
done
step "libero-random" python -u scripts/local/libero_probe.py \
  --ckpt outputs/FINAL_SSV2/ckpt.pt --random_init --label random \
  --out $L/libero_random.json

# 3. entangled control: identical config, one shared trunk feeding both heads. Needed
#    by the LIBERO control row and by CKA; no SSv2 entangled checkpoint existed.
step "train-entangled" python -u scripts/local/train_memory.py \
  --cache_dir outputs/cache/ssv2_W17_full --dataset ssv2 --out_dir outputs/ENT_SSV2 \
  --decoder basedelta --mem_update none --chunk_size_lat 5 --n_chunks 4 \
  --epochs 60 --batch_size 16 --lr 3e-4 \
  --d_static 576 --static_grid 8 --d_dyn 64 --dyn_grid 8 \
  --enc_hidden_ch 192 --dec_hidden_ch 384 \
  --static_target video_median --lambda_static_tgt 1.0 --lambda_consist 3.0 \
  --shared_trunk --num_workers 6 --preload --eval_every 10 --seed 0

step "libero-entangled" python -u scripts/local/libero_probe.py \
  --ckpt outputs/ENT_SSV2/ckpt.pt --label entangled --out $L/libero_entangled.json

echo "QUEUE_OK @ $(date -Is)"
