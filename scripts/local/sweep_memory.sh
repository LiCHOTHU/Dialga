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

# --- A: memory across chunks (static camera, grid decoder) ---
run A1_base            --mem_update none --mem_collapse mean
run A4_gru             --mem_update gru  --mem_collapse mean
run A5_attn            --mem_update attn --mem_collapse mean

# --- B: base+delta decoder -- what actually gives z_static a job ---
#   prescreen: zs_cost 17%->62%, swap 36%->94%; costs reconstruction.
run B1_basedelta       --mem_update none --mem_collapse mean --decoder basedelta
run B2_basedelta_gru   --mem_update gru  --mem_collapse mean --decoder basedelta

# --- C: moving camera. The 3D-memory question, across the whole taxonomy ---
#   implicit  = learned state, no geometry           (gru)
#   explicit  = world-frame canvas, pose in/pose out (canvas)
#   hybrid    = explicit registration + learned gate (gru_world, canvas_gru)
run C1_pan_none        --synth_pan --mem_update none   --mem_collapse mean
run C2_pan_gru         --synth_pan --mem_update gru    --mem_collapse mean
run C3_pan_gru_world   --synth_pan --mem_update gru    --mem_collapse world --d_pose 32
run C4_pan_canvas      --synth_pan --mem_update canvas --mem_collapse mean  --d_pose 32
run C5_pan_canvas_gru  --synth_pan --mem_update canvas_gru --mem_collapse mean --d_pose 32

# --- D: the full proposal -- explicit scene memory AND a decoder that needs it ---
run D1_canvas_bd       --synth_pan --mem_update canvas     --mem_collapse mean \
                       --d_pose 32 --decoder basedelta
run D2_canvas_gru_bd   --synth_pan --mem_update canvas_gru --mem_collapse mean \
                       --d_pose 32 --decoder basedelta

echo "SWEEP_DONE"
