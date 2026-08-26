#!/usr/bin/env bash
# PHASE E: combine what phases B/C separated.
#
# B/C findings this is built on:
#   * static RATE buys nothing on CLEVRER -- attr mAP flat 0.719-0.747 from 96 to 768
#     floats, and 96 has the best recon. So stay at 96/4x4 and stop paying for rate.
#   * InfoNCE is the ONLY term producing semantics: off -> mAP 0.743->0.684 and the
#     stationary/moving gap collapses to +0.009, while zs_cost jumps to 159% and recon
#     IMPROVES. Load-bearing and semantic are in tension and InfoNCE picks the second.
#   * lambda_indep is expensive: indep=0 gives the best recon of the whole sweep
#     (0.0081, -21% vs baseline) and costs almost no mAP (0.739 vs 0.743).
#   * the median/mean teacher gives the best static-vs-dynamic semantic GAP (+0.079
#     vs +0.052 baseline) for a recon cost.
# Nobody has combined indep-off with the teacher, or pushed InfoNCE by WEIGHT rather
# than by batch size (batch 64 hurt recon badly, so more negatives is not the lever).
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/overnight; mkdir -p "$OUT"
BASE="--cache_dir outputs/cache/clevrer_W33_10k --epochs 30 --preload --max_videos 2000 \
      --eval_every 10 --seed 0 --batch_size 16 --d_static 96 --static_grid 4"
TGT="--static_target video_median --lambda_static_tgt 1.0"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $BASE --out_dir "$d" "$@" \
    && touch "$d/DONE" || echo "[FAIL] $n"; }
run E1_i0_tgt      --lambda_indep 0.0 $TGT
run E2_i03_tgt     --lambda_indep 0.3 $TGT
run E3_i0_tgtmean  --lambda_indep 0.0 --static_target video_mean --lambda_static_tgt 1.0
run E4_i0_nce2     --lambda_indep 0.0 --lambda_consist 2.0
run E5_i0_nce2_tgt --lambda_indep 0.0 --lambda_consist 2.0 $TGT
run E6_i0_nce4_tgt --lambda_indep 0.0 --lambda_consist 4.0 $TGT
run E7_i0_tgt_vid  --lambda_indep 0.0 $TGT --mem_update video
echo "PHASEE_DONE"
