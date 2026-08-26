#!/usr/bin/env bash
# HEAD-TO-HEAD of the memory mechanisms as the z_static builder, all at the SAME
# budget (384 floats, 8x8 grid) so what is being compared is the mechanism.
#
#   H2   attention-pooled video-level code          (what I reported: 0.0728)
#   H8   PatchMemory, pose-free                     MosaicMem retrieve-and-compose,
#                                                   alignment left to attention
#   H9   PatchMemory + median-mosaic teacher
#   H11  PatchMemory with KNOWN pose                both alignments live
#                                                   (warped-latent + warped-RoPE)
#   H10  WorldCanvasMemory with known pose          MUSt3R/Spann3R-style accumulation
#                                                   into a world buffer larger than
#                                                   the view
# SSv2 has no camera pose, so H10/H11 add a synthetic pan to supply one; H8/H9 are the
# honest pose-free setting. The canvas NEEDS pose (its warps are identity without it),
# which is why there is no pose-free canvas arm.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/seed_sweep; mkdir -p "$OUT"
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
        --epochs 30 --batch_size 16 --preload --max_videos 2000 --eval_every 10 \
        --d_static 384 --static_grid 8"
TGT="--static_target video_median --lambda_static_tgt 1.0"
FAILED=0
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py $COMMON --out_dir "$d" "$@"; then
    touch "$d/DONE"; else echo "[FAIL] $n"; FAILED=$((FAILED+1)); fi; }
S=0
run H8_s$S  --seed $S --mem_update patch
run H9_s$S  --seed $S --mem_update patch $TGT
run H11_s$S --seed $S --mem_update patch  --synth_pan --d_pose 32
run H10_s$S --seed $S --mem_update canvas --synth_pan --d_pose 32
[ "$FAILED" -gt 0 ] && exit 1
echo "MEMTYPE_DONE"
