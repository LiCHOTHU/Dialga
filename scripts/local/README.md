# Local (single-GPU) experiments — static scene memory for `z_static`

Everything here runs on the **local RTX 5090 box**, not the cluster. Cluster jobs
still live in `scripts/sbatch/`.

## Environment

```bash
conda create --name dialga --clone river   # torch 2.7.0+cu128, sm_120 (Blackwell)
conda activate dialga && pip install av    # torchvision.io video decoding
```

`river` already carried a Blackwell-capable torch, so cloning it beats building from
scratch. Wan-2.2 TI2V-5B weights are already in `~/.cache/huggingface/hub`.

Local data: CLEVRER at `datasets/CLEVRER/train_video` (10,000 videos, 128 frames each)
with annotations at `datasets/CLEVRER/annotations`. LIBERO-90 (4,500 demos, mp4 +
action npy) sits at `/home/licho/libero_90_processed` — unused so far; its episodes are
short (~71 frames ≈ 2 chunks), so CLEVRER is the primary testbed.

## Why this exists

`z_static` is recomputed from scratch for every chunk and thrown away, so it has no
way to be consistent over a video. Measured on the raw Wan latents
(`diag_static_memory.py`, 150 videos):

| | lag 1 | lag 2 | lag 3 |
|---|---|---|---|
| drift of the per-chunk static estimate (rel-MSE) | 0.083 | 0.154 | 0.187 |

and for a *learned* per-chunk code it is far worse (0.379 / 0.573 / 0.753). Robust
collapse does not fix it (median 0.0840 vs mean 0.0834): inside a 33-frame chunk an
object barely moves so a median keeps it; between chunks it has moved. Only
accumulation across chunks can.

A second measurement says the static code is barely used at all. Deleting it costs
~10% reconstruction while deleting `z_dyn` costs ~43% — `z_dyn` is a full-resolution
per-frame 8×8 grid (2304 floats vs 96), so it re-encodes whole frames and leaves
`z_static` no job.

## Pipeline

```bash
# 1. cache CLEVRER -> Wan latents, 4 consecutive chunks per video (W=33 -> T_lat=9)
scripts/local/run_with_restart.sh "[done]" outputs/logs/cache_W33_10k.log \
    scripts/local/cache_clevrer_W33.sh          # ~10.4 windows/s, 40k windows, 5.3 GB

# 2. zero-training diagnostic on the cache
python scripts/local/diag_static_memory.py --cache_dir outputs/cache/clevrer_W33_10k

# 3. the arm ladder (waits for the cache, resumes per arm)
scripts/local/run_with_restart.sh "SWEEP_DONE" outputs/logs/mem_sweep.log \
    scripts/local/sweep_memory.sh

# 4. read it
python scripts/local/summarize_sweep.py
```

`run_with_restart.sh` relaunches until a sentinel appears — long background jobs on
this box get silently SIGKILL'd, and every inner script is resumable.

## What the pieces are

- `src/data/clevrer_sequence.py` — one item = one **video** (K chunks in order), so a
  memory can be carried along it. Optional `preload` holds the cache in RAM as fp16.
- `src/model/static_memory.py` — two independent levels:
  **within-chunk collapse** (`mean` / `median` / `sweep` / `world`, the last two being
  the existing pose-conditioned MosaicMem aggregators) × **across-chunk update**
  (`none` / `ema` / `gru` / `attn`, the last CUT3R-style). All updates are causal, so
  the model runs online at constant cost per chunk (the MUSt3R property).
- `src/model/memory_encoder.py` — DIALGA's encoder run over a whole video. Pose is
  relativised to the **video's** first frame, not each chunk's, so every chunk's grid
  lands in one common reference (the DUSt3R move).
- `src/model/base_delta_decoder.py` — `x̂_t = Base(z_static) + [Δ(z_dyn_t) − mean_t Δ]`.
  The delta branch never sees `z_static` and is exactly zero-mean after the
  nonlinearity, so `mean_t x̂ = Base(z_static)` identically: the static code is the
  only thing that can produce the time-constant part. (Zero-meaning the *code* instead
  does not work — a nonlinear decoder recovers constant structure from it.)
- `synthetic_pan_sequence` in `src/model/camera_pose.py` — pans one continuous camera
  across the whole video, so chunks see different parts of the scene and memory has
  something real to accumulate. The existing `synthetic_pan` restarts per chunk.

## Reading the sweep

`zs_cost` (recon penalty for deleting `z_static`) and `swap` (penalty for using another
video's `z_static`) are the load-bearing columns: they say whether the static code is
doing a job at all. `drift1/3` and `retain0` say whether it is video-length consistent.
`still_zs` vs `move_zd` tests the original intuition — stationary objects should read
from `z_static`, moving ones from `z_dyn`.
