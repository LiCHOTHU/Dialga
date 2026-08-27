#!/usr/bin/env bash
# PHASE W: make the recommendation defensible.
#   1. THREE SEEDS of the leading candidate. Every swap number so far is single-seed and
#      the margins are ~0.05; tonight I had to retract several single-seed rankings.
#   2. The ENTANGLED CONTROL (--shared_trunk: one conv trunk feeding both heads). The
#      swap test only becomes evidence if a model that CANNOT factorize fails it.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/final2; mkdir -p "$OUT"
B="--cache_dir outputs/cache/clevrer_W33_10k --preload --max_videos 2000 --eval_every 60 \
   --batch_size 16 --epochs 60 --lambda_indep 0 --lambda_consist 3 \
   --static_target video_median --lambda_static_tgt 1.0 --static_grid 8 \
   --d_static 768 --d_dyn 768"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $B --out_dir "$d" "$@" && touch "$d/DONE" || echo "[FAIL] $n"; }
for S in 0 1 2; do
  run BD_s$S      --seed $S --decoder basedelta                    # the candidate
  run GRID_s$S    --seed $S                                        # same rate, no constraint
  run ENTANG_s$S  --seed $S --decoder basedelta --shared_trunk     # entangled control
done
echo "PHASEW_DONE"
