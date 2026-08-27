#!/usr/bin/env bash
# PHASE I: validate the winner properly. Two gaps in what the sweep can claim:
#   1. G8 reads -30% recon at 60 epochs but is compared against a 30-EPOCH baseline.
#      That is not a fair number until the baseline gets 60 epochs too.
#   2. Every arm is one seed.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/final; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 20 \
   --batch_size 16 --d_static 96 --static_grid 4 --epochs 60"
MED="--static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
for S in 0 1 2; do
  run BASE60_s$S  --seed $S                                        # the missing control
  run WIN60_s$S   --seed $S --lambda_indep 0 --lambda_consist 3 $MED
  run WIND60_s$S  --seed $S --lambda_indep 0 --lambda_consist 3 $MED \
                  --dino_cache_dir outputs/cache/dino_clevrer_W33 \
                  --dino_to static --lambda_dino 0.1
done
echo "PHASEI_DONE"
