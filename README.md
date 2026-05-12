# DIALGA

DIALGA is a video representation learning project. The aim is to learn a latent representation of CLEVRER video that is:

- **object-aware** — per-slot decomposition with per-slot visibility flags for objects that enter or leave the camera frame,
- **physics-aware** — shaped by `AccelNet` + symplectic Verlet integration (antisymmetric pairwise forces, Newton's-3rd-law by construction),
- **event-aware** — `EventHead` predicts per-(frame, slot) collision probability from $(q, v, a, z^{\text{static}})$, supervised by CLEVRER's ground-truth `collision_mask`.

## Motivation — from the structure of human memory

The architecture is shaped by one principle distilled from cognitive science: **prediction error is the unified encoding signal at every timescale**. Memory is not a generic content store — it factors into complementary systems because *what is surprising* is different from *what is regular*, and the two need different machinery.

We carry that factoring across to video state:

| Cogsci system | DIALGA analog | What it carries |
|---|---|---|
| Semantic / world-model memory (smooth, predictable, *no prediction error*) | Continuous slot state $(q, v, a, z^{\text{static}})$ + `AccelNet` + Verlet | Object identity, position, smooth trajectories |
| Episodic / event memory (sparse, surprising, *high prediction error*) | `EventHead` collision logits | Discrete contact events with participants and timing |

Continuous dynamics handles the part of the world that obeys Newton smoothly. `EventHead` handles the part that doesn't — impulsive collisions, where momentum changes faster than `AccelNet` can integrate. This isn't a wetware analogy ("build a hippocampus module"); it's the computational primitive — sparse, error-driven event encoding alongside a dense state model. Texture and habituated motion auto-discard because they don't produce error; surprises survive.

This framing motivates two concrete choices in the code:

1. **`EventHead` is supervised by `collision_mask`, not by reconstruction loss.** Collisions are the *event* signal; treating them as a separate supervised channel keeps them from being averaged away by per-pixel objectives.
2. **The event head input is $[q, v, a]$ (not just $q$).** $a = q^{t+1} - 2q^t + q^{t-1}$ is the Newton's-1st-law residual — exactly the *prediction error* of the inertial model. Spikes in $a$ are where the smooth dynamics fails; that is what events are.

## What the representation is evaluated on

Probes, in order of what they test:

1. **Reconstruction** — render slot state to pixels through a frozen Wan-2.2 VAE latent decoder.
2. **Forward-prediction** — roll the slot state through `AccelNet`+Verlet, then decode.
3. **Counterfactual** — edit the slot state (delete a slot, change `z_static`), re-rollout, decode.
4. **Collision-event** — extract events from `EventHead` logits, compare to GT collisions by P/R/F1 with a 2-frame tolerance.

Pixel-perfect prediction is *not* the headline; representation quality is.

Detailed per-run results in `RESULTS.md`.

## Code organization

| File | Role |
|---|---|
| `scripts/train_slot.py` | Two-stage Hydra trainer: state + AccelNet + event head, then pixel decoder. |
| `scripts/train_event_head_only.py` | Frozen-encoder event-head trainer (isolates the head from encoder co-adaptation). |
| `scripts/test_event_head.py` | Phase-3 collision-event evaluation (P/R/F1 vs GT). |
| `scripts/eval_overfit_per_video.py` | Per-video state / rollout / decoder MSE (rungs 4 + 5). |
| `scripts/eval_rollout_decode.py` | Rung 6: full pipeline rollout-then-decode. |
| `scripts/eval_counterfactual.py` | Rung 7: delete-slot counterfactual probe. |
| `scripts/eval_state_overlay.py` | Visualize predicted slot positions over GT video. |
| `scripts/overfit_wan_flow.py` / `eval_wan_flow.py` / `probe_wan_vae.py` | Wan-flow decoder training / eval / VAE ceiling. |
| `src/model/accel_net.py` | `AccelNet` + `verlet_step`. |
| `src/model/event_head.py` | `EventHead`, `build_event_features`, `dilate_label`. |
| `src/model/slot_lagrangian.py` | `SlotQueryEncoder`, `CollisionImpulse`, `SlotPixelDecoder`. |
| `src/model/wan_flow_decoder.py` | DiT-style Wan-VAE flow-matching decoder. |
| `src/data/clevrer_paired.py` | `(frames, state, collision_mask)` paired dataset. |
| `src/dynamics/events.py` | Event extraction + `compare_events_to_gt`. |
| `conf/config_slot.yaml` | Single config file. |

## Math

### Slot state

For each video window of $T$ frames the encoder produces, per slot $i$:

- positions $q_i^t \in \mathbb{R}^2$, with finite-difference velocity $v_i^t = (q_i^{t+1} - q_i^{t-1})/2$ and acceleration $a_i^t = q_i^{t+1} - 2 q_i^t + q_i^{t-1}$,
- a learned static identity $z_i^{\text{static}} \in \mathbb{R}^{d_s}$ (video-pooled with visibility weighting),
- visibility $\alpha_i^t \in \{0, 1\}$.

### AccelNet + Verlet

`AccelNet` is a small MLP that maps slot state to per-slot acceleration with two structural priors:

- **Antisymmetric pairwise force.** $f_{ij} = \tfrac{1}{2}(f_\theta(s_i, s_j) - f_\theta(s_j, s_i))$, enforcing Newton's 3rd law by construction.
- **Symplectic Verlet integration.** $q_i^{t+1} = 2 q_i^t - q_i^{t-1} + a_i^t \, \Delta t^2$, energy-stable over long rollouts.

`CollisionImpulse` is enabled in Stage 2 for the impulsive contacts that smooth `AccelNet` cannot represent.

### EventHead

A per-slot 1D-temporal CNN. With `event_input_mode=qva`:

$$
x_i^t \;=\; [\,q_i^t,\; v_i^t,\; a_i^t,\; z_i^{\text{static}}\,]
$$

Supervised by the CLEVRER ground-truth `collision_mask`, dilated by `event_label_dilation` frames for slack around the contact moment, with BCE and `pos_weight = event_pos_weight` (50, for the ~2 % positive rate):

$$
\mathcal{L}_{\text{event}}
 \;=\; \frac{1}{|\mathcal{M}|}\sum_{(i,t) \in \mathcal{M}} \mathrm{BCEWithLogits}\!\big(\ell_i^t,\; \mathrm{dilate}(c_i^t)\big)
$$

with $\mathcal{M}$ the per-slot validity mask. Feeding $v, a$ explicitly is what made the head work: a `q`-only head plateaued at F1 ≈ 0.05 because it had to learn finite differencing under encoder noise; `qva` gets F1 = 0.78 on the 20-vid overfit.

### Stage-1 objective

$$
\mathcal{L}_{\text{stage1}} \;=\;
   \lambda_{\text{state}} \mathcal{L}_{\text{state}}
 + \lambda_{\text{solver}} \mathcal{L}_{\text{solver}}
 + \lambda_{\text{static}} \mathcal{L}_{\text{static}}
 + \lambda_{\text{event}} \mathcal{L}_{\text{event}}
$$

- $\mathcal{L}_{\text{state}}$: encoder positions vs annotated $q$.
- $\mathcal{L}_{\text{solver}}$: one-step Verlet MSE (encoder $q$ → `AccelNet`+Verlet → encoder $q$ at next frame).
- $\mathcal{L}_{\text{static}}$: $z^{\text{static}}$ consistency across visible frames.
- $\mathcal{L}_{\text{event}}$: dilated BCE above.

**Loss balancing matters.** 300-vid runs need $\lambda_{\text{state}} \approx 5$ — otherwise the BCE event term ($\approx 1.4$) drowns the state term ($\approx 0.2$) and the encoder fails to bind slots to objects.

### Stage-2 objective

Encoder + dynamics + event head are frozen. The Wan-flow decoder learns a velocity field $u_\phi$ via linear-path flow matching in Wan-VAE latent space:

$$
\mathcal{L}_{\text{decode}} = \mathbb{E}_{t,\,z_0,\,c} \Big\| u_\phi\big((1-t) z_0 + t z_1,\, t,\, c\big) \;-\; (z_1 - z_0) \Big\|_2^2
$$

where $c$ is the per-slot conditioning bundle (position, velocity, $z^{\text{static}}$, visibility, frame embedding, slot embedding) plus the encoded first frame as a reference latent.

State-and-dynamics learning is by state-space objectives only; pixel rendering is by pixel-space objectives only.

## Results so far

(See `RESULTS.md` for the full record.)

### 5-video overfit ladder (slot pipeline, AccelNet + Verlet)

Pixel MSE expressed as multiples of the Wan-VAE round-trip ceiling (6.0e-5):

| Rung | What | Result |
|---|---|---|
| 1 | Wan-VAE round-trip ceiling | 6.0e-5 (floor) |
| 2 | Wan-flow + GT $q$ + GT attrs | 1.04× ceiling |
| 3 | Wan-flow + encoder $q$ + learned $z^{\text{static}}$ | 1.18× ceiling |
| 4–5 | Encoder regression + 1-step / full-window dynamics rollout | RMSE on $q$ stays tight; open-system handling at slot entry frames |
| 6 | Rollout-then-decode | 1.60× ceiling (mean across 5 videos) |
| 7 | Counterfactual edit (delete a slot, re-rollout, re-decode) | 0.000000 pixel diff on the 4 untouched videos; 0.0037 on the targeted video |

### 20-video EventHead overfit (qva input mode)

In-distribution Phase-3: **F1 = 0.779**, P = 1.000, R = 0.638 (TP = 30, FP = 0, FN = 17).

- 100 % precision is the strongest signal: when the head fires, it is right.
- `qva` was the unlock: a `q`-only head plateaued at F1 ≈ 0.05 on the same data.

### 300-video joint run

The latest run with $\lambda_{\text{state}} = 5$ reached `state_loss = 0.034` at epoch 25/60 before walltime — the encoder converges cleanly under the rebalanced loss. The follow-up combines `event_input_mode=qva` + $\lambda_{\text{state}} = 5$ + a longer walltime to close the loop on 300 videos.

## Configuration — `conf/config_slot.yaml`

| Field | Meaning |
|---|---|
| `model.event_input_mode` | `qva` (default): concat $[q, v, a]$ → `3 × num_state_dims` channels to the head. |
| `model.event_hidden`, `event_kernel`, `event_depth` | `EventHead` shape. |
| `model.d_static` | dim of $z^{\text{static}}$. |
| `dataset.max_objects`, `image_size`, `pos_normalize` | scene + coordinate setup. |
| `training.window_length`, `windows_per_video`, `max_videos` | sampling. |
| `training.stage1_epochs`, `stage2_epochs`, `batch_size` | schedule. |
| `training.lambda_state` | encoder vs GT $q$ — set to 5 in scaling runs. |
| `training.lambda_solver`, `lambda_static` | 1-step Verlet and $z^{\text{static}}$ consistency. |
| `training.lambda_event`, `event_pos_weight`, `event_label_dilation` | event-head supervision (50, 3). |
| `training.lambda_collision` | Stage-2 `CollisionImpulse`. |
| `training.noise_sigma` | GNS-style position-noise injection (0 in overfit, ~5e-3 in scaling). |

## Setup

Conda env `river` (Python 3.12, PyTorch 2.6+cu124, `diffusers>=0.30` for `AutoencoderKLWan`).

Internal path: `/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python`.

Data:

- Videos: `/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video/`
- Annotations: `/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations/train/` (fetched by `scripts/download_clevrer_annotations.sh`)

## Running

```bash
# Stage 1+2: encoder + AccelNet + event head, then encoder-frozen pixel decoder
python scripts/train_slot.py \
    training.lambda_event=1.0 \
    model.event_input_mode=qva \
    training.lambda_state=5.0

# Phase-3 collision-event evaluation
python scripts/test_event_head.py \
    --ckpt output/<run>/stage1.pt \
    --in_distribution \
    --output output/<run>/phase3_indist.json

# Frozen-encoder event-head trainer
python scripts/train_event_head_only.py \
    --src_ckpt output/<src_run>/stage1.pt \
    --output_dir output/<new_run> \
    --use_diff_features --ckpt_every 5

# Per-video state / rollout / decoder MSE
python scripts/eval_overfit_per_video.py --ckpt output/<run>/stage2.pt

# Wan-flow probe pipeline
python scripts/probe_wan_vae.py --ckpt output/<run>/stage1.pt --out_dir outputs/wan_vae_probe
python scripts/overfit_wan_flow.py \
    --ckpt_encoder output/<run>/stage1.pt --out_dir outputs/wan_flow_run \
    --cond_source z_static --use_i0 --use_gt_pos --ema_decay 0.999 --time_dist logitnorm
python scripts/eval_wan_flow.py \
    --ckpt_decoder outputs/wan_flow_run/decoder.pt \
    --ckpt_encoder output/<run>/stage1.pt --steps_list 8,16,32,64
```

Slurm launchers live in `scripts/sbatch_*.sh`. The active event-head launchers are:

- `sbatch_5_overfit_eventhead.sh` — 5-vid overfit sanity.
- `sbatch_20_overfit_eventhead.sh` — 20-vid qva architecture validation.
- `sbatch_300_eventhead_v2.sh` — 300-vid joint run with $\lambda_{\text{state}} = 5$.
- `sbatch_300_eventhead_frozenenc.sh` — frozen-encoder head-only on a pre-converged 300-vid encoder.
