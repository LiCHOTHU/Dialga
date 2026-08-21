# DIALGA

**Plug-in disentanglement of a frozen video-VAE latent.**

Modern video VAEs (Wan-2.2, and predictive variants like PV-VAE) compress video
into rich but **entangled** latents: identity, object motion, and camera motion are
fused into one code that cannot be read, edited, or predicted factor by factor.
Recovering that structure normally means *training a new VAE from scratch*. DIALGA is
a lightweight adapter that re-encodes the latent of **any frozen** video VAE into
named, independently accessible slots — a spatial static code (identity), a per-frame
spatial dynamics code (object motion), and an optional camera code — **without
retraining the VAE**.

Decoding, forward prediction, and counterfactual editing are treated as *probes* of
the factorization, not as the objective.

## What we claim (and don't)

We are explicit about the regime. At extreme compression (96–352 floats/chunk,
distilled from a frozen VAE latent) DIALGA does **not** beat web-pretrained encoders
(DINOv2/VideoMAE/VideoFlexTok) on semantic accuracy, nor the frozen VAE on
reconstruction — we report both plainly. The contribution is the **decomposition
itself**, and the properties it enables, measured against the *fair* peer class
(methods on the same frozen latent, matched rate):

- **Efficient encoding (Table: decodability).** DIALGA's code retains **94.6%** of the
  full-latent reconstruction quality at ~11× compression — ahead of VideoMAE /
  VideoFlexTok (77%) and DINOv2 (61%).
- **Label efficiency (Table: Q1b).** At 360 labels its 96-float code reads attributes
  at **0.854 mAP**, above a PCA of the same latent (0.814) *and* the full 27,648-float
  latent (0.819) — more sample-efficient than the latent it is distilled from.
- **Spatial-dynamics reconstruction fix (Table: Q4a).** Giving `z_dyn` a spatial axis
  raises held-out DROID reconstruction by **+4.4 dB**; a matched-rate ablation
  attributes it to spatial *structure* (+2.7 dB), not added rate (+0.6 dB).
- **Factorization / camera-awareness.** A diagonal-dominant cross-probe (identity from
  `z_static`, motion from `z_dyn`), and a known-pose path for viewpoint stability that
  entangled and camera-blind baselines lack.

Honest open item: the static/dynamic separation is **partial** — identity still leaks
into `z_dyn` (a shared-trunk entangled baseline is nearly as clean), so the
"disentanglement beats entangled" claim needs an independence-loss retrain to become
airtight. See `DEVLOG.md`.

## Architecture

```
RGB video chunk (33 frames, 128×128)
        │
        ▼   frozen Wan-2.2 VAE (TI2V-5B)
Wan latent (48, 9, 8, 8)
        │
        ▼   LatentEncoder3D  (src/model/latent_encoder.py)
   z_static  (spatial grid, ~96 floats)      → identity
   z_dyn     (per-frame spatial grid, ~256)  → object motion   [--dyn_spatial]
   z_cam     (optional)                      → camera motion   [--use_camera_pose]
        │
        ▼   SpatialGridDecoder (src/model/latent_decoder.py)
Wan latent reconstruction
```

Training (`scripts/train_v5.py`) uses a latent reconstruction loss + InfoNCE
consistency + a light forward-prediction term (`--lambda_fwd`, the predictive
component) + optional DINOv2-feature distillation (`--lambda_mae`). No pixel loss
enters the encoder. Key flags:
`--pool_type spatial --dyn_spatial --dyn_grid 8` (the spatial factorization),
`--shared_trunk` (the entangled-AE ablation), `--use_camera_pose --static_agg`
(camera path).

## Repository layout

```
DEVLOG.md                       # dated per-experiment log (paper-ready)
iclr2026/                       # the paper (main.tex + sections/, LaTeX)
scripts/
  train_v5.py                   # main training entry (all datasets)
  cache_wan_latents.py          # CLEVRER  → Wan latents
  cache_wan_ssv2.py / _ucf101.py# SSv2 / UCF101 → Wan latents
  cache_dino_patch.py / _ssv2.py# DINOv2 patch-feature cache (for --lambda_mae)
  extract_droid_wrist.py        # DROID wrist-camera latents + pose
  probes/                       # all evaluation probes (see below)
  sbatch/                       # SLURM launchers (embers-qos, bad-node guarded)
src/
  model/latent_encoder.py       # LatentEncoder3D (static/dyn/camera slots)
  model/latent_decoder.py       # SpatialGridDecoder
  model/camera_pose.py          # known-pose aggregation (PlaneSweep/WorldMemory)
  data/clevrer_window.py        # CLEVRER paired-chunk dataset
  data/{ssv2,droid,libero}_window.py
```

### Key probes (`scripts/probes/`)

| Probe | Fills | What it measures |
|---|---|---|
| `baseline_probe_table.py`      | semantic mAP | z_static vs PCA/mean/full-latent/random/DINOv2 |
| `semantic_efficiency.py`       | rate + label curves | more-semantic-per-float, label efficiency |
| `clevrer_semantic_probe.py`    | cross-probe | identity from z_static vs leakage into z_dyn |
| `clevrer_decode_baselines.py`  | decodability | latent-MSE of each frozen rep + ablation |
| `clevrer_baselines_probe.py`   | pretrained refs | VideoMAE/VideoFlexTok/DINOv2 on CLEVRER |
| `ssv2_action_probe.py`         | SSv2 top-1 | motion read-out (+ full-latent ceiling) |
| `rollout_eval.py`              | forward dynamics | z_dyn rollout vs copy-last |

## Quick start

```bash
conda activate river          # training / probes  (LaTeX: env `tex`, tectonic)

# 1. Cache Wan-VAE latents (one-time)
python scripts/cache_wan_latents.py --data_dir <CLEVRER> --out_dir <CACHE> --max_videos 10000

# 2. Train the factorized adapter on the frozen latent
python scripts/train_v5.py --dataset clevrer --cache_dir <CACHE> --out_dir <OUT> \
    --pool_type spatial --decoder_type spatial --dyn_spatial --dyn_grid 8 \
    --d_static 96 --d_dyn 256 --lambda_mae 0.5 --lambda_fwd 0.1

# 3. Probe (example: semantic efficiency vs the fair peer class)
python scripts/probes/semantic_efficiency.py --ckpt <OUT>/v5_best.pt \
    --cache_dir <CACHE> --out results/semantic_efficiency.json
```

Cluster jobs go through `scripts/sbatch/*.sbatch` (SLURM, `--qos=embers`, guarded
against the known bad-ECC node). Environment: Python 3.12, PyTorch 2.6+cu124,
`diffusers` 0.36 (`AutoencoderKLWan`); Wan-2.2 weights from
`Wan-AI/Wan2.2-TI2V-5B-Diffusers`.

Per-experiment results and the full method/diagnosis history land in `DEVLOG.md`.
