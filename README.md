# DIALGA

Discrete Integration And Lagrangian Generative Architecture.

## Problem

This repo studies a simple question:

- can we learn a latent state from CLEVRER videos,
- predict its future evolution,
- and then test whether adding Lagrangian structure improves prediction quality?

The important lesson from the current code and experiments is that these are really **two separate goals**:

1. learn a latent video dynamics model that actually predicts future frames,
2. test whether Lagrangian structure helps once the latent representation is already usable.

The repo now reflects that split.

## Current Status

The latent-video path is the main active path.

Current findings from the finished runs:

- Best overall predictor so far: `lewm_patch + direct_predictor + identity`
- Best structured model so far: `lewm_patch + lagrangian + identity`
- `wan_vae` is now treated as an ablation path rather than the main direction.
- `SIGReg` is implemented correctly, but it is not currently the main source of improvement.

So the practical recommendation is:

- use `direct_predictor` for debugging and overfitting checks,
- use `lagrangian` when testing whether structure helps,
- keep `lewm_patch` as the main encoder,
- keep `identity` as the default representation.

## Main Pipelines

### 1. Latent-video pipeline

Main file:

- `train.py`

Main components:

- `src/model/lewm_autoencoder.py`
  - learned patch-based encoder/decoder
- `src/model/direct_predictor.py`
  - simple residual next-latent predictor
- `src/model/lagrangian_net.py`
  - structured latent Lagrangian model
- `src/model/state_representation.py`
  - optional residual projector + `SIGReg`
- `src/data/clevrer_sequence.py`
  - short contiguous video-window dataset

### 2. Object-state pipeline

Files:

- `scripts/train_dynamics.py`
- `scripts/eval_rollout.py`
- `src/data/clevrer_states.py`
- `src/dynamics/`

This is still the cleaner physics path conceptually, but it depends on CLEVRER annotations being available locally.

## Current Model Options

### Encoder (`model.latent_source`)

- `lewm_patch`
  - recommended main encoder now
- `wan_vae`
  - frozen WAN VAE ablation
- `dino_vits14`
  - experimental frozen DINOv2 path

### Dynamics model (`model.dynamics_model`)

- `direct_predictor`
  - recommended simplest baseline
- `lagrangian`
  - structured model to test whether the Lagrangian inductive bias helps

### Representation (`model.representation`)

- `identity`
  - recommended default
- `projected_sigreg`
  - optional trainable residual projection regularized by `SIGReg`

`identity` means the encoder output is used directly as the latent state.

## Math

### Latent state

For an observation frame `o_t`, the encoder produces a latent map:

$$
z_t = E(o_t)
$$

If `representation=identity`, then

$$
q_t = z_t
$$

If `representation=projected_sigreg`, then

$$
q_t = S_\psi(z_t) = z_t + r_\psi(z_t)
$$

where `r_psi` is a small residual adapter.

### Direct predictor

The simplest predictor learns:

$$
\hat q_{t+1} = P_\phi(q_{t-1}, q_t)
$$

Its training objective is dominated by predictive losses:

$$
\mathcal{L}_{\text{pred}} = \|\hat q_{t+1} - q_{t+1}\|_2^2
$$

and a decoded image-space prediction term:

$$
\mathcal{L}_{\text{pred-recon}} = \|D(\hat q_{t+1}) - o_{t+1}\|_2^2
$$

If the encoder is trainable, we also use a standard reconstruction term:

$$
\mathcal{L}_{\text{recon}} = \|D(q_t) - o_t\|_2^2
$$

### Lagrangian model

The structured latent model learns a discrete Lagrangian between adjacent latent states:

$$
L_\theta(q_{t-1}, q_t) = T_\theta(q_t - q_{t-1}) - V_\theta\left(\frac{q_{t-1}+q_t}{2}\right)
$$

The current head predicts:

- one positive scalar mass from the CLS token,
- additive patch-wise potential over midpoint features.

The discrete Euler-Lagrange residual is:

$$
R_\theta(q_{t-1}, q_t, q_{t+1}) = D_2 L_\theta(q_{t-1}, q_t) + D_1 L_\theta(q_t, q_{t+1})
$$

and the DEL regularizer is:

$$
\mathcal{L}_{\text{DEL}} = \|R_\theta(q_{t-1}, q_t, q_{t+1})\|_2^2
$$

In the current practical setup, DEL is used as a **weak structural regularizer**, not the main learning signal.

### Total latent objective

The current trainer uses some subset of:

$$
\mathcal{L} =
\lambda_{\text{solver}}\mathcal{L}_{\text{pred}} +
\lambda_{\text{pred-recon}}\mathcal{L}_{\text{pred-recon}} +
\lambda_{\text{recon}}\mathcal{L}_{\text{recon}} +
\lambda_{\text{DEL}}\mathcal{L}_{\text{DEL}} +
\lambda_{\text{sigreg}}\operatorname{SIGReg}(Z)
$$

where `SIGReg` is the isotropy / anti-degeneracy regularizer on random 1D latent projections.

## Recommended Settings

### Recommended debug baseline

Use this when asking: “can the latent stack predict future frames at all?”

- `model.latent_source=lewm_patch`
- `model.dynamics_model=direct_predictor`
- `model.representation=identity`

### Recommended structured experiment

Use this when asking: “does Lagrangian structure help compared with the direct predictor?”

- `model.latent_source=lewm_patch`
- `model.dynamics_model=lagrangian`
- `model.representation=identity`

### WAN path

Treat `wan_vae` as an ablation. It is still usable, but it is no longer the main recommended encoder.

## Setup

### Environment

The repo expects at least:

- `torch`
- `torchvision`
- `hydra-core`
- `omegaconf`
- `wandb`
- `imageio`
- `timm`
- `transformers`

If you want to run the WAN VAE path, `src/model/autoencoder.py` expects the WAN VAE import from `actaim`.

It looks for a sibling workspace:

```text
../ActAIM3
```

and imports:

```python
from actaim.models.wan.vae.vae2_2 import Wan2_2_VAE
```

### Data

Default CLEVRER video path:

```text
/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video
```

Object-state experiments additionally require CLEVRER annotation JSON files.

## Running The Code

### 1. Overfit sanity check on a very small set

Recommended first check:

```bash
python train.py \
  hydra.run.dir=output/overfit2_direct_local \
  wandb.enabled=false \
  training.device=cuda \
  training.epochs=120 \
  training.lr=0.0005 \
  training.lr_warmup_epochs=0 \
  training.use_scheduler=false \
  training.sequence_window_length=6 \
  training.sequence_windows_per_video=1 \
  training.sequence_max_videos=2 \
  training.train_subset_size=2 \
  training.val_fraction=0.0 \
  training.batch_size=1 \
  training.num_workers=0 \
  training.log_interval=5 \
  training.inference_every=5 \
  training.inference_every_steps=0 \
  model.latent_source=wan_vae \
  model.dynamics_model=direct_predictor \
  model.representation=identity \
  training.lambda_del=0.0 \
  training.lambda_solver_mse=1.0 \
  training.lambda_pred_recon=5.0
```

This saves to:

```text
output/overfit2_direct_local/
```

### 2. Main LeWM direct baseline

```bash
python train.py \
  hydra.run.dir=output/lewm_direct \
  model.latent_source=lewm_patch \
  model.dynamics_model=direct_predictor \
  model.representation=identity \
  training.lambda_del=0.0 \
  training.lambda_solver_mse=1.0 \
  training.lambda_recon=1.0 \
  training.lambda_pred_recon=5.0
```

### 3. Main LeWM Lagrangian experiment

```bash
python train.py \
  hydra.run.dir=output/lewm_lagrangian \
  model.latent_source=lewm_patch \
  model.dynamics_model=lagrangian \
  model.representation=identity \
  training.lambda_del=0.1 \
  training.lambda_solver_mse=1.0 \
  training.lambda_recon=1.0 \
  training.lambda_pred_recon=5.0
```

### 4. Slurm launch

Generic launcher:

```bash
sbatch --job-name="lewm" scripts/train_dialga.sbatch \
  'hydra.run.dir=output/${oc.env:SLURM_JOB_NAME}-${oc.env:SLURM_JOB_ID}' \
  "model.latent_source=lewm_patch" \
  "model.dynamics_model=direct_predictor" \
  "model.representation=identity" \
  "wandb.name=lewm-direct"
```

The repo intentionally keeps only the generic launcher now. One-off experiment-specific Slurm scripts were removed to keep the repo smaller and easier to maintain.

### 5. Object-state path

Train:

```bash
python scripts/train_dynamics.py
```

Evaluate:

```bash
python scripts/eval_rollout.py \
  evaluation.checkpoint_path=/path/to/checkpoint.pt
```

## Outputs

Each latent run writes to its Hydra output directory, typically under `output/...`.

Expected contents:

- `checkpoints/`
- `inference/`
- `.hydra/`
- WandB run files

Important logged metrics:

- `train/total`
- `train/solver_mse`
- `train/recon`
- `train/pred_recon`
- `val/total`
- `val/solver_mse`
- `val/recon`
- `val/pred_recon`

Important media keys in WandB:

- `media/predicted_comparison`
- `media/predicted_rollout`

## Current Caveats

- `lewm_patch + direct_predictor` is currently the strongest predictive baseline.
- `lewm_patch + lagrangian` is the strongest structured model so far, but it still has not clearly beaten the direct predictor on prediction quality.
- `SIGReg` is currently an ablation, not a recommended default.
- `wan_vae` is now an ablation path, not the mainline encoder.
- The object-state pipeline is still the cleaner physics path conceptually, but it requires annotation files that may not be present in every environment.
- `src/perception/sam2_tracker.py` is still a stub.

## Repo Layout

```text
train.py                      # main latent-video trainer
scripts/train_dialga.sbatch   # generic Slurm launcher
scripts/overfit_video.py      # legacy simple overfit script
scripts/train_dynamics.py     # object-state trainer
scripts/eval_rollout.py       # object-state evaluator
scripts/probe_encoder.py      # probing utility
src/model/                    # encoders, dynamics models, state projector
src/data/                     # CLEVRER loaders
src/dynamics/                 # object-state dynamics modules
tests/                        # physics/integrator unit tests
```

## Citation

If you use this repository, please cite the corresponding work or internal project notes that match your experiments.
