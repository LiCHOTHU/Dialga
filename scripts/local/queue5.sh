#!/usr/bin/env bash
# Retrain CLEVRER at the COMMITTED configuration, 3 seeds. tab:matched is the paper's
# weakest table: single-seed, and no surviving checkpoint on disk matches the committed
# base+delta config at its rate (133 CLEVRER checkpoints, none of them this one).
# Rate check: floats = d_static + d_dyn * T_lat = 576 + 64*9 = 1152 = 24.0x on the
# (48,9,8,8) CLEVRER latent -- exactly the rate the table reports, and the same shape
# rule as the committed SSv2 model (576 + 64*5 = 896).
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga:/home/licho/workspace/Dialga/ml-videoflextok
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
while pgrep -f queue4.sh >/dev/null; do sleep 60; done
for s in 0 1 2; do
  echo "=== clevrer-committed seed $s @ $(date -Is) ==="
  python -u scripts/local/train_memory.py \
    --cache_dir outputs/cache/clevrer_W33_10k --dataset clevrer \
    --out_dir outputs/FINAL_CLEVRER_s$s --chunk_size_lat 9 \
    --decoder basedelta --mem_update none --n_chunks 4 \
    --epochs 60 --batch_size 16 --lr 3e-4 \
    --d_static 576 --static_grid 8 --d_dyn 64 --dyn_grid 8 \
    --enc_hidden_ch 192 --dec_hidden_ch 384 \
    --static_target video_median --lambda_static_tgt 1.0 --lambda_consist 3.0 \
    --num_workers 6 --preload --eval_every 10 --seed $s \
    || echo "!!! seed $s failed rc=$?"
done
echo "QUEUE5_OK @ $(date -Is)"
