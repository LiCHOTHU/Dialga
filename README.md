# DIALGA

Object- and physics-aware video **representation** learning on CLEVRER.

The headline claim is about the representation, not the renderings. Decoding,
forward prediction, and counterfactual editing are probes that test whether
the representation actually carries identity, dynamics, and causal events.

---

## Problem

A video of bouncing CLEVRER objects contains three kinds of information that
should live in different places in a good representation:

1. **Identity** — color / material / shape of each object. Constant over time.
2. **Dynamics** — per-frame position and motion of each object. Smooth except
   at events.
3. **Events** — discrete moments where dynamics change (collisions, entries,
   exits). Sparse in time.

Existing video models tend to entangle these. A pixel-reconstruction objective
will happily memorize episode-specific surface statistics — the model "knows
the video" without learning anything transferable across videos. We want a
factorization where each axis is independently accessible: identity-only
swaps, frozen-dynamics rollouts, and event localization should all be cheap
read-outs from the latent state.

## Hypothesis

A bottlenecked encoder that splits its output into three named slots —
`z_static`, `z_dyn`, `event_logits` — and a decoder structurally prevented
from cheating across them, is enough to produce that factorization
**without** auxiliary losses tying each slot to ground-truth state. At
sufficient data scale, the model will choose to use each slot for its
intended role because no other path lets the decoder reconstruct.

Concretely we predict:

- A linear/MLP probe on `z_static` recovers color, material, and shape on
  **held-out videos**, well above the majority-class baseline.
- Freezing `z_dyn` on a held-out video produces a decoded video with no
  detectable motion variation (counterfactual locality).
- `event_logits` fires at GT collisions on held-out videos.

These are testable disentanglement properties, not just reconstruction
quality.

## Method

### Representation

For a `T`-frame video the encoder emits three tensors per `K`-slot batch:

| Tensor | Shape | Role |
|---|---|---|
| `z_static` | `(B, K, 16)` | per-slot identity, constant over time |
| `z_dyn` | `(B, T, K, 32)` | per-slot dynamics state per frame |
| `event_logits` | `(B, T, K)` | per-slot event presence per frame |

A visibility envelope `α[t, k] = cumsum_t softmax_t(event_logits[:, k])`
turns the event channel into a soft mask over time — slots can "enter" but
not exit, in line with CLEVRER physics.

### Architecture

```
RGB video (T, 3, 128, 128)
        │
        ▼   (frozen Wan-2.2 VAE, 705M params)
Wan latent (48, T_lat, 8, 8)
        │
        ▼   TrajectoryEncoder (1.8M params)
  z_static, z_dyn, event_logits
        │
        ▼   TrajectoryDecoder (2.4M params, time-blinded)
Wan latent reconstruction
```

Two architectural details do the load-bearing work:

- **Time-blinded decoder.** No temporal positional embedding on the output
  queries. The decoder cannot know *when* it is querying unless the
  information comes through `z_dyn`. This forces motion to live in `z_dyn`
  rather than leaking into `z_static`.
- **Block-diagonal cross-attention.** Each output query attends only to its
  own temporal slice of slot tokens. Information about frame `t` cannot
  enter the reconstruction of frame `t'`. This *architecturally* enforces
  per-frame dynamics exclusivity; we verified by freezing `z_dyn` at
  inference and observing zero latent variation.

### Losses

```
L = recon                                    # Wan-latent MSE
  + 0.10 · ‖Δ² z_dyn · α‖²                   # smoothness
  + 0.01 · H(event_logits)                   # event sparsity
  + 0.01 · VICReg(z_static)                  # identity variance/covariance
  + 0.02 · event NLL (first-visible frame)   # weak event supervision
```

No pixel-level photometric loss enters the encoder. The decoder is the only
path that touches pixels, and it is symmetric — so the encoder is free to
allocate capacity wherever the latent objective rewards.

### Pipeline

1. **Cache stage** — `scripts/cache_wan_latents.py` encodes CLEVRER videos
   into Wan-VAE latent windows (48 channels × 3 latent frames × 8×8 spatial),
   stored as `<idx>.pt` blobs alongside per-window metadata
   (positions, attrs, slot_mask, collisions). Resume-safe.
2. **Train stage** — `scripts/train_trajectory.py` consumes the cache, splits
   by `video_id`, and trains the encoder/decoder pair end-to-end with random
   PV-VAE-style frame masking.
3. **Probe stage** — separate scripts evaluate each disentanglement claim on
   the held-out split.

## Experiments

### Iter 21 — does the recipe generalize?

The 500-video Iter 18 model fit the training set well but probed *below
chance* on val for identity. Iter 21 tests whether 20× more videos closes
the train/val gap with **no other recipe change**.

**Setup.** 10,000 train videos × 4 windows = 40,000 windows. 80/20 split by
`video_id` (8,000 / 2,000 videos). 60 epochs, batch=4, lr=5e-4 cosine→0,
dropout 0.1. Single H200, ~3 h wall-clock.

**Results.**

| Run | Train recon | Val recon | Ratio | Verdict |
|---|---:|---:|---:|---|
| Iter 18 (500 vids) | 0.0090 | 0.0216 | 2.40× | overfit |
| **Iter 21 (10k vids)** | **0.0104** | **0.0078** | **0.76×** | val < train |

The train/val ratio inverts. At 10k scale, dropout and weight-decay are not
doing the regularization — the data is. This is a necessary condition for
the disentanglement claim but not sufficient: a model could still reconstruct
well by encoding everything into `z_dyn` and ignoring `z_static`.

### Probes (in progress)

The recon generalization above is a precondition. The actual hypothesis tests
are the four probes below. Identity is the load-bearing one.

| # | Probe | Question | Pass criterion |
|---|---|---|---|
| 1 | Identity (`probe_iter21_identity.py`) | Does `z_static` encode color/material/shape on held-out videos? | val_acc clearly above majority baseline (color: chance 12.5%) |
| 2 | Event localization | Do `event_logits` fire at GT collisions on val? | F1 above trivial baseline |
| 3 | Counterfactual | Freeze `z_dyn` on a val video → no decoded motion? | latent Δ ≈ 0 (verified architecturally; needs scale test) |
| 4 | Val GIFs + PSNR | Are reconstructions visually faithful at scale? | qualitative + PSNR vs Wan-VAE ceiling |

Per-experiment results land in `DEVLOG.md` as they finish.

## Repository layout

```
DEVLOG.md                          # per-experiment log, paper-ready
scripts/
  cache_wan_latents.py             # CLEVRER → Wan latents (cache builder)
  train_trajectory.py              # main training entry
  probe_iter21_identity.py         # probe 1: z_static → color/material/shape
  run_iter21_cache_h200.sh         # cluster runner: cache stage
  run_iter21_train_h200.sh         # cluster runner: train stage
  download_clevrer{,_annotations}.sh
src/
  model/trajectory_encoder.py      # TrajectoryEncoder + Decoder + loss
  data/clevrer_paired.py           # paired window dataset (frames + state)
  data/clevrer_states.py           # vocabs (color/material/shape) + state ds
```

## Quick start

```bash
# 1. Cache Wan-VAE latents (one-time, ~70 min on H200 for 10k videos)
bash scripts/run_iter21_cache_h200.sh

# 2. Train (auto-launches when cache marker appears, ~3 h on H200)
bash scripts/run_iter21_train_h200.sh

# 3. Probe identity on val z_static
python scripts/probe_iter21_identity.py \
    --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W12 \
    --ckpt outputs/iter21_10000vid_<stamp>/trajectory.pt \
    --val_frac 0.2 --seed 42
```

Environment: Python 3.12, PyTorch 2.6+cu124, `diffusers` 0.36 (for
`AutoencoderKLWan`). Wan-2.2 weights pulled from
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`.
