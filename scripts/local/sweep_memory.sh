#!/usr/bin/env bash
# Static-memory arm ladder. Waits for the Wan cache to finish, then runs each arm
# in turn. Every arm resumes from its own ckpt.pt, so the whole sweep is safe to
# kill and restart (this box SIGKILLs long background jobs).
#
#   A  static camera, grid decoder  -> does cross-chunk MEMORY fix drift/retention?
#   B  base+delta decoder           -> does forcing the split give z_static a job?
#   C  panning camera               -> does memory pay off when chunks see different
#                                      parts of the scene, and does KNOWN POSE help?
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh
conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga

CACHE=outputs/cache/clevrer_W33_10k
CLOG=outputs/logs/cache_W33_10k.log
OUT=outputs/mem_sweep
EPOCHS="${EPOCHS:-60}"
BS="${BS:-16}"
mkdir -p "$OUT"

# wait for the cache builder to print its completion sentinel
while ! grep -q "^\[done\]" "$CLOG" 2>/dev/null; do
  echo "[wait] cache: $(tail -1 "$CLOG" 2>/dev/null | tr -d '\n')"
  sleep 120
done
echo "[wait] cache complete."

run () {  # run <name> <extra args...>
  local name="$1"; shift
  local dir="$OUT/$name"
  if [ -f "$dir/DONE" ]; then echo "[skip] $name"; return; fi
  echo "=================== ARM $name @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py \
      --cache_dir "$CACHE" --out_dir "$dir" \
      --epochs "$EPOCHS" --batch_size "$BS" --preload "$@" \
    && touch "$dir/DONE"
}

# --- A: does memory across chunks help? (static camera, grid decoder) ---
run A1_base          --mem_update none --mem_collapse mean
run A4_gru           --mem_update gru  --mem_collapse mean
run A3_ema           --mem_update ema  --mem_collapse mean
run A5_attn          --mem_update attn --mem_collapse mean
run A2_median        --mem_update none --mem_collapse median

# --- B: does the base+delta decoder give z_static a real job? ---
run B1_basedelta     --mem_update none --mem_collapse mean --decoder basedelta
run B2_basedelta_gru --mem_update gru  --mem_collapse mean --decoder basedelta

# --- C: moving camera, where a scene memory should actually pay ---
run C1_pan_none      --synth_pan --mem_update none --mem_collapse mean
run C2_pan_gru       --synth_pan --mem_update gru  --mem_collapse mean
run C3_pan_gru_world --synth_pan --mem_update gru  --mem_collapse world --d_pose 32

echo "SWEEP_DONE"
