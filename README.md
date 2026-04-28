# DIALGA

Discrete Integration And Lagrangian Generative Architecture.

## Overview

This repository studies a simple but important question:

- can we learn a compact latent **state** from CLEVRER videos,
- can we learn **forward dynamics** in that state,
- and does adding **Lagrangian structure** improve prediction once the latent itself is usable?

The current codebase is intentionally simplified around a **two-stage pipeline**:

1. **State-and-dynamics training**
   - learn the latent state encoder together with a forward dynamics model
   - optionally regularize the latent with a weak DEL term and/or `SIGReg`
   - do **not** train the pixel decoder in this stage
2. **Decoder training**
   - freeze the learned latent encoder
   - train only the decoder to map latent states back to pixels

This avoids the previous failure mode where physics losses and pixel losses competed for the same latent representation in the same optimization step.

## Current Status

### What currently works best

From the recent experiments:

- **Best overall predictive model:** `lewm_patch + direct_predictor`
- **Best structured model:** `lewm_patch + lagrangian`

So the current workflow should be:

- use `direct_predictor` as the simplest debugging / overfitting baseline,
- use `lagrangian` as the structured comparison model,
- keep the learned `lewm_patch` encoder as the default latent representation.

### What has been de-emphasized

- frozen `wan_vae` paths are no longer the mainline path
- DINO-based paths are experimental and not part of the default workflow
- `SIGReg` is currently an ablation / stability term, not the main reason the model improves

## Main Files

### Stage 1: State and dynamics

- `train.py`
  - trains the latent encoder and a dynamics model in latent space
- `src/model/lewm_autoencoder.py`
  - LeWM-style patch encoder / decoder
- `src/model/direct_predictor.py`
  - simple latent next-state predictor
- `src/model/lagrangian_net.py`
  - latent Lagrangian model
- `src/model/state_representation.py`
  - `SIGReg` only
- `src/data/clevrer_sequence.py`
  - sequence-window dataset from CLEVRER video files

### Stage 2: Decoder training

- `scripts/train_decoder.py`
  - trains only the pixel decoder on top of a frozen latent encoder from Stage 1

### Object-state path

- `scripts/train_dynamics.py`
- `scripts/eval_rollout.py`
- `src/data/clevrer_states.py`
- `src/dynamics/`

This path remains available for explicit state-space physics experiments, but it depends on CLEVRER annotations being available locally.

## The Two-Stage Training Logic

The core design principle now is:

- **state and dynamics should be trained by state-space objectives**
- **pixel rendering should be trained by pixel-space objectives**

This means the code is strict about which modules are updated in each stage.

### Stage 1 updates

Updated:

- latent encoder `E`
- dynamics model `P` or `L`

Frozen:

- decoder `D`

### Stage 2 updates

Updated:

- decoder `D`

Frozen:

- latent encoder `E`
- dynamics model

## Math

### Latent state

For a frame `o_t`, the learned encoder produces a latent map:

$$
q_t = E_\psi(o_t)
$$

The current main encoder is the LeWM-style patch encoder in `src/model/lewm_autoencoder.py`.

### Direct predictor

The simple baseline dynamics model learns:

$$
\hat q_{t+1} = P_\phi(q_{t-1}, q_t)
$$

The Stage 1 loss for the direct predictor is built from latent prediction terms:

$$
\mathcal{L}_{\text{teacher}} = \|P_\phi(q_{t-1}, q_t) - q_{t+1}\|_2^2
$$

and short autoregressive rollout loss:

$$
\mathcal{L}_{\text{rollout}} = \|\hat q_{t+1} - q_{t+1}\|_2^2
$$

where `hat q_{t+1}` is produced by feeding the model's own previous predictions back into itself over a short window.

### Lagrangian model

The structured latent model learns a discrete Lagrangian between adjacent states:

$$
L_\theta(q_{t-1}, q_t) = T_\theta(q_t - q_{t-1}) - V_\theta\left(\frac{q_{t-1}+q_t}{2}\right)
$$

The current implementation predicts:

- one positive scalar mass
- a summed patch-wise potential

The discrete Euler-Lagrange residual is:

$$
R_\theta(q_{t-1}, q_t, q_{t+1}) = D_2 L_\theta(q_{t-1}, q_t) + D_1 L_\theta(q_t, q_{t+1})
$$

and the DEL regularizer is:

$$
\mathcal{L}_{\text{DEL}} = \|R_\theta(q_{t-1}, q_t, q_{t+1})\|_2^2
$$

In the current version, DEL is used as a **weak structural regularizer**, not as the main learning signal.

### SIGReg

`SIGReg` regularizes the latent state distribution through random 1D projections:

$$
\operatorname{SIGReg}(Q) = \frac{1}{M} \sum_{m=1}^{M} T(Qu^{(m)})
$$

where `T` is the Epps–Pulley statistic.

In this repo, `SIGReg` is used only during **Stage 1** and only when explicitly enabled.

### Stage 1 objective

The state-and-dynamics trainer uses a weighted combination of:

$$
\mathcal{L}_{state} =
\lambda_{solver} \mathcal{L}_{rollout} +
\lambda_{DEL} \mathcal{L}_{DEL} +
\lambda_{sigreg} \operatorname{SIGReg}(Q)
$$

For the direct predictor, `lambda_DEL` is ignored.

### Stage 2 objective

Decoder training uses only:

$$
\mathcal{L}_{decode} = \|D(E(o_t)) - o_t\|_2^2
$$

This is trained separately so the latent state does not have to satisfy physics and rendering constraints at the same time.

## Recommended Settings

### Recommended debug baseline

Use this first when asking:

> can the learned latent state support video prediction at all?

- `model.latent_source=lewm_patch`
- `model.dynamics_model=direct_predictor`

### Recommended structured experiment

Use this once the direct baseline is working and you want to test structure:

- `model.latent_source=lewm_patch`
- `model.dynamics_model=lagrangian`

## Configuration

Main config file:

- `conf/config.yaml`

Important fields:

### Model

- `model.latent_source`
  - current supported main value: `lewm_patch`
- `model.dynamics_model`
  - `direct_predictor`
  - `lagrangian`
- `model.lewm_patch_size`
- `model.lewm_embed_dim`
- `model.lewm_latent_channels`
- `model.lewm_encoder_depth`
- `model.lewm_num_heads`

### Training

- `training.sequence_window_length`
- `training.sequence_windows_per_video`
- `training.sequence_max_videos`
- `training.train_subset_size`
- `training.lambda_solver_mse`
- `training.lambda_del`
- `training.lambda_sigreg`
- `training.use_scheduler`

## Setup

### Environment

Required packages include:

- `torch`
- `torchvision`
- `hydra-core`
- `omegaconf`
- `wandb`
- `imageio`

The current main path no longer depends on WAN VAE or DINOv2.

### Data

Default CLEVRER video root:

```text
/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video
```

## Running The Code

### Stage 1: train latent state and dynamics

#### Direct predictor baseline

```bash
python train.py \
  hydra.run.dir=output/lewm_direct \
  training.device=cuda \
  model.latent_source=lewm_patch \
  model.dynamics_model=direct_predictor \
  training.lambda_solver_mse=1.0 \
  training.lambda_del=0.0 \
  training.lambda_sigreg=0.0
```

#### Lagrangian experiment

```bash
python train.py \
  hydra.run.dir=output/lewm_lagrangian \
  training.device=cuda \
  model.latent_source=lewm_patch \
  model.dynamics_model=lagrangian \
  training.lambda_solver_mse=1.0 \
  training.lambda_del=0.1 \
  training.lambda_sigreg=0.0
```

### Stage 2: train the decoder only

After Stage 1 finishes, train the decoder on the frozen state encoder:

```bash
python scripts/train_decoder.py \
  --state_ckpt output/lewm_direct/checkpoints/epoch_010.pt \
  --output_dir output/lewm_decoder \
  --epochs 20 \
  --lr 1e-3 \
  --batch_size 32 \
  --device cuda
```

This produces:

- `output/lewm_decoder/best_decoder.pt`

### Slurm launch

Generic latent state training launcher:

```bash
sbatch --job-name="lewm" scripts/train_dialga.sbatch \
  'hydra.run.dir=output/${oc.env:SLURM_JOB_NAME}-${oc.env:SLURM_JOB_ID}' \
  "model.latent_source=lewm_patch" \
  "model.dynamics_model=direct_predictor" \
  "wandb.name=lewm-direct"
```

The repo intentionally keeps only this generic launcher instead of many one-off experiment launchers.

### Object-state path

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

Each Stage 1 run writes to its Hydra output directory under `output/...`.

Expected contents:

- `checkpoints/`
- `.hydra/`
- `wandb/`

Important logged metrics:

- `train/total`
- `train/anchor_mse`
- `train/solver_mse`
- `train/del`
- `train/sigreg`
- `val/total`
- `val/anchor_mse`
- `val/solver_mse`
- `val/del`
- `val/sigreg`

Stage 1 currently does **not** train the decoder, so state training should be interpreted in latent-space terms first.

## Current Caveats

- `lewm_patch + direct_predictor` is currently the strongest predictive baseline.
- `lewm_patch + lagrangian` is currently the strongest structured model, but it still has not clearly beaten the direct predictor on raw prediction quality.
- `SIGReg` is currently an ablation, not a recommended default.
- The object-state pipeline is still the cleaner physics path conceptually, but it requires CLEVRER annotations being present locally.
- `src/perception/sam2_tracker.py` is still a stub.

## Repository Layout

```text
train.py                      # Stage 1: train latent state + dynamics
scripts/train_decoder.py      # Stage 2: train decoder only
scripts/train_dialga.sbatch   # generic Slurm launcher for Stage 1
scripts/train_dynamics.py     # object-state trainer
scripts/eval_rollout.py       # object-state evaluator
src/model/lewm_autoencoder.py # LeWM-style encoder / decoder
src/model/direct_predictor.py # simple latent predictor
src/model/lagrangian_net.py   # latent Lagrangian model
src/model/state_representation.py # SIGReg
src/data/clevrer_sequence.py  # sequence-window video dataset
src/dynamics/                 # object-state dynamics modules
```

## Citation

If you use this repository, please cite the corresponding work or internal notes that match your experiments.
