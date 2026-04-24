# DIALGA

Discrete Integration And Lagrangian Generative Architecture.

## What Problem This Repo Solves

The repository studies how to learn dynamics from video with an explicit physics bias instead of treating next-frame prediction as a generic black-box visual mapping.

The core question is:

- can we learn a latent state from CLEVRER videos,
- fit a discrete Lagrangian on that state,
- and use the resulting discrete Euler-Lagrange structure to improve rollout behavior and physical consistency?

The code currently has two related pipelines:

1. A latent-video pipeline in `train.py`.
   This is the main path for learning on CLEVRER video triplets with a frozen WAN VAE encoder, a learnable latent Lagrangian, optional DEL solver supervision, and an optional LeWM-style `SIGReg` state-regularization path.
2. An object-state pipeline in `scripts/train_dynamics.py`.
   This trains directly on CLEVRER annotation trajectories using object positions, velocities, and attributes.

## What The Code Currently Does

### Latent-video pipeline

The latent path uses:

- `src/model/autoencoder.py`: a frozen WAN VAE encoder/decoder wrapper
- `src/model/lagrangian_net.py`: a DiT-style discrete Lagrangian network over latent states
- `src/model/state_representation.py`: an optional residual state projector plus `SIGReg`
- `train.py`: training, validation, inference video export, WandB logging, and checkpointing

The input data is CLEVRER frame triplets `(o_{t-1}, o_t, o_{t+1})` loaded from raw `.mp4` files or extracted frame folders via `src/data/clevrer_dataset.py`.

### Object-state pipeline

The object-state path uses:

- `src/data/clevrer_states.py`: CLEVRER annotation loader
- `src/dynamics/lagrangian.py`: object-wise Lagrangian with learned masses and potentials
- `src/dynamics/integrator.py`: leapfrog integrator
- `scripts/train_dynamics.py`: rollout-MSE training on annotation trajectories
- `scripts/eval_rollout.py`: checkpoint evaluation

This path is useful when you want to debug the mechanics directly in state space without the difficulty of image encoding.

## Math Behind The Current Code

### 1. Frozen video encoder and optional state representation

The latent trainer first encodes each frame with a frozen WAN VAE:

$$
\tilde{q}_t = E_{\text{WAN}}(o_t)
$$

By default, the state is just the frozen latent:

$$
q_t = \tilde{q}_t
$$

If `model.representation=wan_projected_sigreg`, the code inserts a trainable residual projector:

$$
q_t = S_\psi(\tilde{q}_t) = \tilde{q}_t + r_\psi(\tilde{q}_t)
$$

where `r_psi` is a small 1x1-conv residual adapter initialized to zero, so training starts from the baseline identity mapping.

### 2. Discrete Lagrangian model in latent space

The main model in `src/model/lagrangian_net.py` learns a discrete Lagrangian between adjacent latent states:

$$
L_\theta(q_{t-1}, q_t) = T_\theta(q_t - q_{t-1}) - V_\theta\left(\frac{q_{t-1}+q_t}{2}\right)
$$

Internally, the code:

- embeds the midpoint latent with a patch embedding,
- runs a small transformer,
- predicts a positive scalar mass from the CLS token,
- predicts patch-wise potentials from patch tokens,
- forms kinetic, potential, and mechanical energy.

### 3. Discrete Euler-Lagrange residual

Training uses the discrete Euler-Lagrange residual over latent triples:

$$
R_\theta(q_{t-1}, q_t, q_{t+1})
= D_2 L_\theta(q_{t-1}, q_t) + D_1 L_\theta(q_t, q_{t+1})
$$

In the code this is computed by autograd in `calculate_del_residual()`.

The main DEL loss is:

$$
\mathcal{L}_{\text{DEL}} = \|R_\theta(q_{t-1}, q_t, q_{t+1})\|_2^2
$$

### 4. Optional solver-rollout supervision

`train.py` also has an optional differentiable solver update that iteratively adjusts a candidate `q_{t+1}` by descending the squared DEL residual energy. If enabled, it adds a rollout supervision term:

$$
\mathcal{L}_{\text{solver}} = \|\hat{q}_{t+1} - q_{t+1}\|_2^2
$$

where `hat{q}_{t+1}` is the solver-refined next latent.

### 5. Optional LeWM-style `SIGReg`

If `training.lambda_sigreg > 0`, the code regularizes the projected latent state with `SIGReg`:

$$
\operatorname{SIGReg}(Z) = \frac{1}{M}\sum_{m=1}^{M} T(Zu^{(m)})
$$

where:

- `Z` is a time/batch stack of flattened latent states,
- `u^{(m)}` are random unit directions,
- `T` is the Epps-Pulley normality statistic used in the LeWM codebase.

This encourages the learned state representation to stay diverse and approximately Gaussian across random 1D projections.

### 6. Total latent training objective

The current `train.py` objective is:

$$
\mathcal{L}
= \lambda_{\text{DEL}}\,\mathcal{L}_{\text{DEL}}
+ \lambda_{\text{solver}}\,\mathcal{L}_{\text{solver}}
+ \lambda_{\text{sigreg}}\,\operatorname{SIGReg}(Z)
$$

The code also logs a constant-velocity anchor error

$$
\mathcal{L}_{\text{anchor}} = \|2q_t - q_{t-1} - q_{t+1}\|_2^2
$$

but that anchor term is diagnostic only and is not directly optimized.

### 7. Object-state Lagrangian

The object-state path in `src/dynamics/lagrangian.py` uses explicit object positions `q`, velocities `q_dot`, attributes, and masks. Its learned Lagrangian is:

$$
L(q, \dot{q}) = T(q, \dot{q}) - V_{\text{ext}}(q) - V_{\text{pair}}(q)
$$

with:

- learned positive masses from object attributes,
- external per-object potential,
- pairwise interaction potential,
- leapfrog rollout for simulation,
- rollout position MSE as the main training loss.

## Repository Layout

```text
train.py                      # main latent-video trainer
scripts/train_dialga.sbatch   # Slurm launcher for train.py
scripts/overfit_video.py      # single-video latent overfit sanity check
scripts/train_dynamics.py     # object-state trainer
scripts/eval_rollout.py       # object-state evaluator
src/model/                    # WAN encoder, latent Lagrangian, state projector
src/dynamics/                 # object-state dynamics modules
src/data/                     # CLEVRER video and annotation loaders
tests/                        # unit tests for integrator / Lagrangian code
```

## Setup

### Environment

This repo assumes a Python environment with at least:

- `torch`
- `torchvision`
- `hydra-core`
- `omegaconf`
- `wandb`
- `imageio`
- `pytest`

The latent-video path also requires the WAN VAE import from `actaim`.

`src/model/autoencoder.py` looks for a sibling workspace:

```text
../ActAIM3
```

and imports:

```python
from actaim.models.wan.vae.vae2_2 import Wan2_2_VAE
```

So either:

1. keep `ActAIM3` next to this repo, or
2. install `actaim` in the active environment.

### CLEVRER data

The default video path in the latent trainer is:

```text
/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video
```

The object-state pipeline expects CLEVRER annotations under:

```text
$SCRATCH/Dialga/CLEVRER/annotations
```

You can override both through Hydra or script arguments.

### WAN VAE checkpoint

The default WAN checkpoint path is:

```text
/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

Override it with `model.vae_ckpt=...` or `--vae-ckpt ...`.

## Running The Code

### 1. Single-video overfit sanity check

This is the fastest way to verify that the frozen WAN VAE latents are learnable at all.

```bash
python scripts/overfit_video.py \
  --video-index 0 \
  --epochs 200 \
  --output-dir output/video_overfit_00000
```

Outputs include:

- checkpoints
- `comparison.png`
- rollout `.mp4` or `.gif`
- `final_metrics.json`

### 2. Latent-video training locally

Baseline frozen latent representation:

```bash
python train.py \
  hydra.run.dir=output/local-frz \
  wandb.enabled=true \
  model.representation=wan_frozen \
  training.lambda_del=1.0 \
  training.lambda_solver_mse=0.0 \
  training.lambda_sigreg=0.0
```

LeWM-style representation with `SIGReg`:

```bash
python train.py \
  hydra.run.dir=output/local-sigreg \
  wandb.enabled=true \
  model.representation=wan_projected_sigreg \
  training.lambda_del=1.0 \
  training.lambda_solver_mse=0.0 \
  training.lambda_sigreg=0.01 \
  training.sigreg_num_proj=256
```

Projector ablation without `SIGReg`:

```bash
python train.py \
  hydra.run.dir=output/local-proj \
  wandb.enabled=true \
  model.representation=wan_projected_sigreg \
  training.lambda_del=1.0 \
  training.lambda_solver_mse=0.0 \
  training.lambda_sigreg=0.0
```

Hybrid training with solver supervision:

```bash
python train.py \
  hydra.run.dir=output/local-hyb \
  wandb.enabled=true \
  model.representation=wan_frozen \
  training.lambda_del=1.0 \
  training.lambda_solver_mse=0.1 \
  training.lambda_sigreg=0.0
```

### 3. Latent-video training on Slurm

The repo includes `scripts/train_dialga.sbatch`.

Example baseline launch:

```bash
sbatch --job-name="frzdel" scripts/train_dialga.sbatch \
  'hydra.run.dir=output/${oc.env:SLURM_JOB_NAME}-${oc.env:SLURM_JOB_ID}' \
  "model.representation=wan_frozen" \
  "training.lambda_del=1.0" \
  "training.lambda_solver_mse=0.0" \
  "training.lambda_sigreg=0.0" \
  "wandb.group=repr-key" \
  "wandb.name=repr-frz-del-e1"
```

Example `SIGReg` launch:

```bash
sbatch --job-name="sg010d" scripts/train_dialga.sbatch \
  'hydra.run.dir=output/${oc.env:SLURM_JOB_NAME}-${oc.env:SLURM_JOB_ID}' \
  "model.representation=wan_projected_sigreg" \
  "training.lambda_del=1.0" \
  "training.lambda_solver_mse=0.0" \
  "training.lambda_sigreg=0.01" \
  "training.sigreg_num_proj=256" \
  "wandb.group=repr-key" \
  "wandb.name=repr-sg010-del-e1"
```

### 4. Object-state training

Train the annotation-based object-state model:

```bash
python scripts/train_dynamics.py
```

Useful overrides:

```bash
python scripts/train_dynamics.py \
  dataset.annotation_dir=/path/to/annotations \
  dataset.video_dir=/path/to/train_video \
  training.epochs=50 \
  training.batch_size=8
```

Evaluate an object-state checkpoint:

```bash
python scripts/eval_rollout.py \
  evaluation.checkpoint_path=/path/to/checkpoint.pt
```

## Outputs And Logging

### Latent-video path

Each latent run writes to its Hydra output directory, typically under `output/...`.

You should expect:

- `checkpoints/`
- `inference/`
- `.hydra/`
- WandB run files if `wandb.enabled=true`

During long runs, `train.py` now logs:

- step-level `train_step/*`
- running `train_running/*`
- epoch-level `train/*`
- periodic `val/*`
- `media/predicted_comparison`
- `media/predicted_rollout`

Inference media is saved locally and also sent to WandB.

### Object-state path

The object-state trainer writes Hydra outputs and checkpoints under:

```text
$SCRATCH/Dialga/runs/object_dynamics/...
```

## Tests

Run the unit tests with:

```bash
pytest tests
```

These cover:

- integrator behavior
- energy conservation on simple systems
- permutation behavior of the object-state Lagrangian

## Current Caveats

- The latent-video path uses a frozen WAN video VAE on single frames. This works in practice, but it is still a compromise.
- If `model.representation=wan_projected_sigreg`, the inference video is decoded from projected latents, so the reconstruction is approximate and should be treated as a qualitative diagnostic.
- The object-state pipeline is cleaner physically, but it depends on CLEVRER annotations rather than learning state directly from pixels.
- `src/perception/sam2_tracker.py` is still a stub; a full perception-to-state-to-render loop is not yet wired end to end.

## Citation

If you use this repository, please cite the corresponding work or project notes used in your experiments.
