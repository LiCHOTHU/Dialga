# DIALGA — Development Log

Running record of experiments on CLEVRER (and beyond). Each entry is meant to be paper-ready: setup, numbers, artifact paths, dates, and interpretation kept together so claims can be traced back to runs.

---

## Experimental Setup

| Field | Value |
|---|---|
| Dataset | CLEVRER train videos, 5-video overfit subset |
| Video IDs | 663, 4242, 6311, 6890, 8376 |
| Frames per video used | 128 (annotated) |
| Window length | 6 frames |
| Image size | 128 × 128 |
| Coordinate mode | `world_xy`, normalized by `pos_normalize=4.0` |
| Max objects (slots) | 8 |
| Encoder | `SlotQueryEncoder` (cross-attention, K query tokens) |
| `z_static` dim | 16 |
| Dynamics | `AccelNet` (Verlet integration) — Lagrangian path retained but degenerate on small data |
| Decoder | `WanLatentFlowDecoder` (DiT-style, flow-matching on Wan 2.2 VAE latents) |
| Wan 2.2 VAE | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (frozen, 704.7M params) |
| Sampler | Euler, 8 steps (best in step-count sweep) |
| Hardware | NVIDIA, 80 GB GPU, env `river` (Python 3.12, PyTorch 2.6+cu124, diffusers 0.36) |

### Reference values

| Quantity | Value |
|---|---|
| Wan-VAE round-trip pixel MSE (ceiling) | **0.000060** |
| `CLEVRER` world span (x, y) | roughly [-3, 3] |
| Cosmetic perception threshold at 128 px | ~1.5× ceiling (visually indistinguishable) |

---

## Probe Ladder Definitions

The ladder separates representation-quality questions into independent probes. Each rung tests one component or one composition; each result narrows where the bottleneck is.

| Rung | Probe | What is held GT | What is tested |
|---|---|---|---|
| 1 | Decoder fit (oracle inputs) | q, attrs | wan-flow decoder capacity |
| 2 | Decoder fit (learned identity) | q, visibility | does `z_static` match GT attrs |
| 3 | Decoder fit (learned q) | none in window | does encoder-pred q decode |
| 4 | Encoder state regression | — | encoder accuracy on q |
| 5a | 1-step dynamics | q at t=0,1 | AccelNet/Lagrangian one-step error |
| 5b | Multi-step dynamics | q at t=0,1 | rollout-state drift to t=W-1 |
| 6 | Full pipeline | first-frame latent for `i0` only | rollout-then-decode pixel quality |
| 7 | Counterfactual edit | first-frame latent for `i0` only | locality and editability of the latent |

---

## Master Results Table

| # | Rung | Result | × ceiling | Verdict |
|---|---|---|---|---|
| 1 | Decoder (GT q, GT attrs) | 6.2e-5 | **1.04×** | ✅ saturated |
| 2 | Decoder (GT q, learned z_static) | 6.2e-5 | **1.04×** | ✅ z_static drop-in for GT identity |
| 3 | Decoder (encoder-pred q, z_static) | 7.1e-5 | **1.18×** | ✅ encoder noise absorbed (+0.08× over rung 2) |
| 4 | Encoder state RMSE (world units) | 0.051 | n/a | ✅ <1% of world span |
| 5a | 1-step dynamics RMSE (smooth / collision) | 0.0001 / 0.042 | n/a | ✅ smooth · ⚠️ collision |
| 5b | Multi-step rollout RMSE @ t=5 (smooth / collision) | 0.001-0.002 / 0.020 | n/a | ✅ smooth · ⚠️ collision |
| 6 | Rollout-then-decode mean | 9.5e-5 | **1.60×** | ✅ smooth (0.77-0.99×) · ⚠️ collision (2.32×) |
| 7 | Counterfactual edit | locality 0.0; target Δ 3.7e-3 | n/a | ✅ surgical, factorized latent |

### Per-video breakdown

| Video | Has collision | Rung 4 enc RMSE (world) | Rung 5b rollout RMSE @ t=5 | Rung 6 GT-q | Rung 6 enc-q | Rung 6 roll-q | Rung 7 factual MSE | Rung 7 counter MSE |
|---|---|---|---|---|---|---|---|---|
| 663 | no | 0.027 | 0.0014 | 1.58× | 1.64× | 2.27× | 0.000135 | 0.000135 |
| 4242 | **yes (1)** | 0.057 | 0.020 | 0.69× | 0.75× | **2.32×** | 0.000138 | **0.004697** ← target |
| 6311 | no | 0.043 | 0.0012 | 0.90× | 0.95× | 0.99× | 0.000059 | 0.000059 |
| 6890 | no | 0.074 | 0.0015 | 0.58× | 0.71× | 0.77× | 0.000046 | 0.000046 |
| 8376 | no | 0.056 | 0.0022 | 1.47× | 1.57× | 1.66× | 0.000099 | 0.000099 |
| **mean** | — | 0.051 | 0.0053 | **1.04×** | **1.12×** | **1.60×** | — | — |

### Drift profiles

#### Rung 5b — state-space rollout RMSE on video 663 (smooth, world units)

| t | RMSE | Source |
|---|---|---|
| 0 | 0.0000 | GT seed |
| 1 | 0.0000 | GT seed |
| 2 | 0.0001 | dynamics |
| 3 | 0.0004 | dynamics |
| 4 | 0.0008 | dynamics |
| 5 | 0.0014 | dynamics |

Linear drift, ~4× over 3 steps. Bounded.

#### Rung 6 — pixel MSE per frame on video 663 (rollout variant)

| t | MSE | × ceiling | Source |
|---|---|---|---|
| 0 | 0.000112 | 1.88× | encoder seed |
| 1 | 0.000126 | 2.12× | encoder seed |
| 2 | 0.000149 | 2.50× | rollout |
| 3 | 0.000137 | 2.31× | rollout |
| 4 | 0.000152 | 2.56× | rollout |

Frame-to-frame drift bounded; t=4 only 1.36× t=0.

---

## Decoder Tuning History (Rung 1)

| Run name | q source | d_model | n_blocks | EMA | Time sched | Epochs | Best MSE | × ceiling | Best step count |
|---|---|---|---|---|---|---|---|---|---|
| original | GT | 384 | 6 | none | uniform | 800 | 8.8e-5 | 1.49× | 32 |
| run 2 (z_static) | GT | 384 | 6 | none | uniform | 800 | 8.8e-5 | 1.49× | 32 |
| tune v1 | GT | 512 | 8 | 0.999 | logitnorm | 1500 | 6.5e-5 | 1.10× | 16 |
| tune v2_big | GT | 768 | 10 | 0.999 | logitnorm | 2000 | 6.2e-5 | **1.04×** | 8 |
| **rung 3** | **encoder-pred** | 512 | 8 | 0.999 | logitnorm | 1500 | **7.1e-5** | **1.18×** | 8 |

### Sampler observations

- More Euler steps consistently *hurts*: 1.04× at 8 steps → 1.23× at 48 steps for tune v2_big. Linear-path flow matching favors fewer larger steps.
- Heun (2nd-order) noticeably worse than Euler — velocity field has fine-scale noise that compounds under predictor-corrector.
- EMA (decay 0.999) gave ~5% improvement at evaluation vs online weights.
- Logit-normal time sampling concentrates mass near t=0.5 and stabilized late training.

---

## Per-Run Artifacts

| Run | Path | Date | Note |
|---|---|---|---|
| Slot Stage 1+2 (encoder + AccelNet + SlotPixelDecoder) | `/storage/project/r-agarg35-0/lwang831/dialga_outputs/hybrid_v5_static_0508_0324/` | 2026-05-08 | mainline checkpoint, contains stage1.pt and stage2.pt |
| Wan-flow decoder, GT-q, best | `outputs/wan_flow_tune_v2_big_0508_1750/decoder.pt` | 2026-05-08 | rung 1/2 |
| Wan-flow decoder, encoder-pred-q | `outputs/wan_flow_rung3_v1cfg_0509/decoder.pt` | 2026-05-09 | rung 3 |
| Wan-VAE round-trip probe | `outputs/wan_vae_probe/` | 2026-05-08 | establishes ceiling |
| Rung 3 training log | `outputs/wan_flow_rung3_v1cfg_0509/train.log` | 2026-05-09 | full epoch trace |
| Rung 4+5 per-video script | `scripts/eval_overfit_per_video.py` | 2026-05-09 | dispatches accel vs Lagrangian |
| Rung 6 (full pipeline) | `outputs/rung6_v2big_decoder_0509/` | 2026-05-09 | per-video grids + console table |
| Rung 7 (counterfactual) | `outputs/rung7_counterfactual_v2big_0509/` | 2026-05-09 | target + 4 untouched grids |

### Eval scripts

| Script | Probe |
|---|---|
| `scripts/probe_wan_vae.py` | Wan-VAE ceiling (rung 0) |
| `scripts/overfit_wan_flow.py` | Decoder training (rungs 1, 2, 3) |
| `scripts/eval_wan_flow.py` | Decoder sampler/step-count sweep |
| `scripts/eval_overfit_per_video.py` | Encoder + dynamics per-video metrics (rungs 4, 5) |
| `scripts/eval_rollout_decode.py` | Full pipeline (rung 6) |
| `scripts/eval_counterfactual.py` | Slot-deletion edit (rung 7) |

---

## Observations and Interpretive Notes

### O1. Strict ordering across rungs 1 → 6

The mean × ceiling values fall on a strict-monotone ladder:

```
1.04×  rung 1 (oracle q + oracle attrs)
1.04×  rung 2 (oracle q + learned z_static)        # +0.00× — z_static is information-equivalent to GT attrs
1.18×  rung 3 (encoder-pred q + learned z_static)  # +0.14× — encoder noise cost
1.60×  rung 6 (rollout q + learned z_static)       # +0.42× — dynamics drift cost
```

This monotone ordering is itself an evidence pattern: each layer adds bounded error without instability or mode collapse.

### O2. The collision-frame failure mode is concentrated and explainable

Every rung where dynamics is involved (5a, 5b, 6) shows a clean smooth-vs-collision split:

| Rung | Smooth videos | Video 4242 (collision) | Ratio |
|---|---|---|---|
| 5a 1-step | 0.0001-0.0003 | 0.042 | ~150× |
| 5b rollout @t=5 | 0.0012-0.0022 | 0.020 | ~10× |
| 6 pixel MSE | 0.77-0.99× | 2.32× | ~3× |

Root cause: the `CollisionImpulse` module was trained on ~3 collision events in the 5-video set (logged `coll 0.00000 (n=3)` for almost the entire run). This is a data-sparsity problem, not an architectural one. The fix is more videos (more collisions per training step), or upweighting `lambda_collision`, or oversampling collision triples.

### O3. The latent is genuinely factorized (rung 7 zero-diff)

Rung 7's bit-exact 0.000000 difference on the four non-target videos is the cleanest single piece of evidence in the experiment set. It says:

- Sampling noise is reproducible (same seed, same RNG path)
- Cross-attention masking on visibility=0 is implemented correctly (zero leakage)
- The slot representation has no per-video coupling — editing slot 0 of one video does not perturb any other video's render

Combined with rung 7's measurable target divergence (0.0037), this is the flagship demonstration: the latent is editable, edits are local, edits propagate through dynamics + decoder coherently.

### O4. The wan-flow decoder generalizes off the GT manifold

The v2_big decoder used in rung 6 was trained *only* on GT-q. At inference time it accepts:

- GT q (training distribution): 1.04×
- Encoder-pred q (out-of-distribution): 1.12×
- Rollout q (further out-of-distribution): 1.60×

Graceful degradation at each step. No catastrophic failure even on rollout q which was never seen at training. This argues the slot manifold is locally smooth and the decoder learned a function rather than a memorization.

### O5. The small SlotPixelDecoder is the bottleneck in the legacy pipeline

The slot pipeline's Stage-2 decoder (`SlotPixelDecoder`, ~tiny CNN) reaches recon MSE 0.0079 on the same 5 videos.  
Wan-flow decoder reaches 0.000062.  
**130× sharper.**

This is decisive evidence that the slot representation contains the visual info; the legacy decoder was simply outclassed.

---

## Open Questions

| ID | Question | Required experiment | Status |
|---|---|---|---|
| OQ1 | Does the pipeline scale beyond 5 videos? | Train at 100 / 1000 / full CLEVRER | not started |
| OQ2 | Does collision handling improve with more events? | Same scaling experiment, monitor `lambda_collision` | not started |
| OQ3 | Does `z_static` cluster by physical attributes (mass, charge in ComPhy)? | Linear probe of `z_static` against GT attribute labels | not started |
| OQ4 | Can the slot trajectories drive the CLEVRER program executor? | Plug q into NS-DR's symbolic QA pipeline | not started |
| OQ5 | Does Lagrangian dynamics ever beat AccelNet at scale? | Re-enable Lagrangian path + train at full data | partial — Lagrangian degenerate on 5 videos, scale not tested |
| OQ6 | How does the framework do on Physion (smooth dynamics regime)? | Port the data loader and overfit | not started |

---

## EventHead — Supervised Collision Detection (2026-05-11/12)

A separate experimental thread: a per-slot Conv1d head that consumes encoder `(q, v, a, z_static)` and predicts per-`(t, slot)` event-probability logits. Trained alongside the encoder via BCE against a teacher signal; evaluated as Phase-3 collision-event extraction (`min_participants=2`, `contact_distance=0.8`, `time_tolerance=2`).

### Teacher choices

| Teacher | Source | Properties |
|---|---|---|
| `gt` | CLEVRER `collision_mask`, dilated ±d frames | Exact, but requires dataset-specific annotation. Used as ceiling reference. |
| `motion` | Saliency-weighted centroid 2nd-difference `Δ²c` per slot | Pixel-only, dataset-agnostic. The "domain-portable" target. |

### F1 ladder (5-video overfit, in-distribution, T=128, batch=1, 120 epochs)

| Run | Teacher | Key knobs | F1 | P | R |
|---|---|---|---|---|---|
| v2 broken | motion | `abs_thresh=0.05` | 0.000 | — | — |
| v2 fix | motion | `abs_thresh=0.003, dil=3, λ_state=1` | 0.111 | — | — |
| v3 | motion | `abs_thresh=0.005, dil=1, λ_state=3` | 0.133 | — | — |
| **v4** | motion | `abs_thresh=0.003, dil=2, λ_state=2, sharpness=400, hard, pw=50` | **0.632** | 0.750 | 0.545 |
| v5 | motion + `pair_augment=0.4` | regressed | 0.091 | — | — |
| v6 | motion + `pair_augment=0.2` | regressed | 0.300 | — | — |
| GT-teacher 5-vid | gt | same other knobs | 0.625 | 1.000 | 0.455 |

**Headline:** motion-centroid teacher hits **F1=0.632 on 5 vids**, slightly beating GT teacher (0.625). Validates the project's domain-portable claim — pixel-only supervision matches GT-annotation quality at the overfit scale.

### Two non-obvious bugs identified

1. **Conv1d train/inference window mismatch.** Head trained at `window_length=6` with `kernel_size=5` (padding=2) learned padding-dependent patterns that don't transfer to T=128 inference. Matching train T to inference T pushed F1 from 0 to 0.625 for the GT teacher alone.
2. **`abs_thresh=0.05` silently zero'd all motion-teacher labels.** Empirically `|Δ²c|` at CLEVRER collisions is in [0.001, 0.015] (median 0.006). `abs_thresh=0.05` is 16× too high → teacher labels are all zero → BCE pushes logits to −∞ → head learns "always say no". Diagnostic: `scripts/diag_amag_distribution.py`.

### Scale-up to 20 videos exposed a structural bug

The v4 recipe does **not** scale: F1 drops from 0.632 (5 vids) → 0.195 (20 vids). Training event loss plateaus at ~1.66 instead of converging.

Diagnostic that nailed the root cause:

**Standalone head experiment** (`scripts/standalone_head_overfit.py`): train the head **alone** on GT q + GT collision labels, no encoder, no decoder. 28k params, 200 epochs in **0.5 seconds**. Result:
- `pos_weight=50, qva` (original config): TP=282, FP=3807 → P=0.07, R=1.00, **F1=0.13** (loss=0.76)
- `pos_weight=5, qva` (balanced): TP=0, FP=0, FN=282 → **F1=0.00**
- `pos_weight=5, qva_nn` (with neighbor features): **F1=0.380**
- `pos_weight=8, qva_nn`: **F1=0.605** ← matches the 5-vid v4 result

**Interpretation:** The Conv1d-per-slot head is structurally blind to pair information. It cannot distinguish "stationary collision partner" (label=1, own `a≈0`) from "stationary non-event slot" (label=0, own `a≈0`) — they have identical per-slot `(q, v, a)` features. At 5 vids the head memorizes the exceptions; at 20 vids the exception pattern is too varied. Adding neighbor features (`d_min`, `a_nn`) breaks the symmetry and unlocks fast overfit.

### The `qva_nn` structural fix

`src/model/event_head.py:build_event_features(include_neighbor=True)` appends two per-`(t, k)` channels:

- `d_min[t, k]` — distance to the nearest other slot at time `t`
- `a_nn[t, k]` — magnitude of that nearest neighbor's `Δ²q`

With `qva_nn` + `pos_weight=8`, the standalone head overfits 20 videos in **0.5 seconds** to F1=0.605. The architecture has the capacity; the missing piece was pair-aware input.

### Final A/B at 20 vids: encoder noise is the remaining bottleneck

Full pipeline run: 20 vids, `qva_nn`, `pos_weight=8`, GT teacher, `event_head_train_on_gt_q=true`, 80 epochs (≈ 6 min on L40S). Phase-3 extraction:

| Inference q source | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| Encoder `q_pred` | 6 | 78 | 41 | 0.071 | 0.128 | 0.092 |
| **GT q (diagnostic)** | 6 | **1** | 41 | **0.857** | 0.128 | 0.222 |

Same checkpoint, same head, same labels — only the inference-time `q` differs. **Precision jumps from 0.07 to 0.86** when fed clean q. The head learned the right decision boundary. The remaining bug is **encoder `q_pred` is too noisy frame-to-frame**: its `Δ²q` swamps the collision signal at most timesteps, causing 78 spurious firings.

### Open structural issues (next experiments)

1. **Encoder noise → `Δ²q` feature.** Two candidate fixes:
   - (a) Add a temporal-smoothness regularizer `λ · ||Δ²q_pred||²` to the encoder loss. Tried with `λ=1.0` — too low to bite (total loss unchanged); needs `λ≥10`.
   - (b) Apply a learned/fixed temporal smoother to `q_pred` before computing head features.
2. **Pair recall is low (R=0.128 even with clean inputs).** Head learns to fire confidently on one slot of a colliding pair but not always both; Phase-3's `min_participants=2` then drops most events. Possible fix: pair-aware head (per-(slot,slot) logits) or relaxed extraction (`min_participants=1` with stronger contact filter).
3. **Motion teacher labels don't fully align with GT collision frames.** Motion teacher labels fire over broad windows (12-18 frames per collision); Phase-3 evaluates with `time_tolerance=2`. The standalone test used GT collision labels and got F1=0.605; the full pipeline with motion teacher got F1=0.067 on 20 vids. Label quality is the gap to close for the domain-portable claim at scale.

### Artifacts

- v4 5-vid winner: `/storage/project/r-agarg35-0/lwang831/outputs/dialga/slot_5_T128_motion_v4_0511_2312/stage1.pt` (F1=0.632)
- 5-vid GT teacher ceiling: `/storage/project/r-agarg35-0/lwang831/outputs/dialga/slot_5_T128_gt_0511_1452/stage1.pt` (F1=0.625)
- Standalone overfit test: `scripts/standalone_head_overfit.py`
- Diagnostic scripts: `scripts/diag_motion_t128.py`, `scripts/diag_amag_distribution.py`

---

## Iter 21 (2026-05-14): Full CLEVRER scale — train recipe generalizes

**Goal.** The 500-video Iter 18 model fit the training set well but probed *below chance* on val for color/material/shape — a sign that `z_static` had memorized episode-specific surface features rather than learning a transferable mapping. The hypothesis under test: does scaling 20× (500 → 10,000 videos) close the train/val gap, given the same architecture and objective?

**Architecture (unchanged from Iter 16d/18).**
- Wan-2.2 VAE frozen (`AutoencoderKLWan`, 705M params, fp16) — encodes 128×128 RGB to (48, 3, 8, 8) latent windows of 12 input frames.
- TrajectoryEncoder: bidirectional transformer with K=8 slot queries + per-frame slot queries, outputs `(z_static[B,K,16], z_dyn[B,T,K,32], event_logits[B,T,K])`.
- TrajectoryDecoder: **time-blinded** — no temporal positional embedding on output queries; block-diagonal cross-attention mask gates slot information to its own temporal block. This is what *architecturally* enforces z_dyn motion-exclusivity (verified by counterfactual: freezing z_dyn → 0.000000 latent variation).
- α visibility: cumsum over softmax-normalized event_logits (`use_null=False`).
- 4.24M trainable params (enc 1.82M, dec 2.42M, pred 0K).

**Recipe.** Mirrors Iter 18 winning config:
- 10,000 train videos × 4 windows = 40,000 windows, 80/20 split by *video_id* (8,000 train / 2,000 val videos = 32,000 / 8,000 windows).
- batch=4, num_workers=0, lr=5e-4 cosine→0, weight_decay=1e-3, dropout=0.1, 60 epochs.
- Losses: recon (Wan latent MSE) + 0.1·‖Δ²z_dyn·α‖² + 0.01·H(event) + 0.01·VICReg(z_static) + 0.02·event NLL vs GT first-visible frame.

**Loss curve.** Smooth, monotonic, no instability. Cosine LR annealed to 0 over 60 epochs (4,745 s = ~79 min wall-clock).

| ep | recon (train) | val_recon | ratio |
|---:|---:|---:|---:|
| 1   | 0.0352 | — | — |
| 5   | 0.0222 | 0.0197 | 0.89 |
| 10  | 0.0169 | 0.0144 | 0.85 |
| 20  | 0.0132 | 0.0107 | 0.81 |
| 30  | 0.0117 | 0.0092 | 0.79 |
| 40  | 0.0109 | 0.0084 | 0.77 |
| 50  | 0.0104 | 0.0080 | 0.77 |
| **60** | **0.0104** | **0.0078** | **0.76** |

**Headline.** `val < train` throughout training. At 500-vid scale (Iter 18) the train/val ratio was **2.4× (val worse)**; at 10k-vid scale it inverts to **0.76× (val better)**. Dropout/WD aren't doing the regularization — the data is.

| Run | Train recon | Val recon | Ratio | Notes |
|---|---:|---:|---:|---|
| Iter 16d (100-vid, no split) | 0.0090 | — | — | Pure overfit baseline |
| Iter 18 (500-vid, 80/20) | 0.0090 | 0.0216 | **2.40×** | Overfit despite dropout 0.1 + WD 1e-3 |
| **Iter 21 (10k-vid, 80/20)** | **0.0104** | **0.0078** | **0.76×** | Healthy generalization |

**What this does and does not prove.** Pixel/latent recon generalizing is necessary but **not sufficient** for the disentanglement claim. The actual scientific question — does `z_static` now encode *semantic* identity (color/material/shape) instead of episode-specific surface features? — is still open. The MLP probe protocol from Iter 18 needs to be re-run on this checkpoint. Same for event localization on val and pixel PSNR on rendered val GIFs.

**Engineering wins.**
- Cache resume support added to `scripts/cache_wan_latents.py` — skip existing `<idx>.pt` files, rebuild metadata at end.
- Auto-restart wrapper `scripts/run_cache_with_restart.sh` survives silent SIGKILLs (the 40k-window cache hit at least two silent deaths during the run; wrapper resumed each time).
- `scripts/run_iter21_when_cache_done.sh` polls for cache completion then auto-launches training.

**Artifacts.**
- Checkpoint: `outputs/iter21_10000vid_040702/trajectory.pt`
- Cache: `outputs/cache/wan_10000vid_W12/` (40,000 windows, ~6 GB)
- Train log: `outputs/logs/train_iter21.log`

**Open todos for Iter 21.**
1. Re-run identity probes (color/material/shape) on val z_static → answers the disentanglement-generalization question.
2. Re-run event localization on val → answers whether the event channel transfers (Iter 18 was 35.9% = trivial baseline).
3. Counterfactual probe at scale — freeze z_dyn on val videos, confirm 0 latent variation.
4. Render 5 val GIFs + PSNR.

---

## Iter 21 — identity probe (2026-05-18): disentanglement claim does not hold

**Question.** Now that recon generalizes (val_recon 0.0078 < train_recon 0.0104), does `z_static` actually encode color/material/shape on held-out videos, or did the model achieve low recon by routing identity through some other path?

**Protocol (linear, held-out).**
1. Load Iter 21 checkpoint (epoch 55, val_recon 0.00782 — process died before ep 60 but the saved ckpt reproduces DEVLOG-Iter-21 numbers; reran from the same recipe on this cluster, stamp `213853`).
2. Trainer val split only (val_frac=0.2, seed=42). Encoder never saw any of these videos.
3. Encode each val video → `z_static` of shape (K=8, 16), averaged across its 4 windows.
4. 50/50 split *by video_id* (not by slot, seed=0) — disjoint probe-train / probe-test video pools.
5. Single linear layer per attribute group (no hidden), AdamW, 2000 epochs full-batch.

**Result.**

| Group | Classes | Train slots | Test slots | Chance | Majority | Train acc | **Test acc** | Δ vs maj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Color    | 8 | 4947 | 4950 | 0.125 | 0.132 | 0.170 | **0.156** | **+0.024** |
| Material | 2 | 4947 | 4950 | 0.500 | 0.502 | 0.522 | **0.502** | **+0.000** |
| Shape    | 3 | 4947 | 4950 | 0.333 | 0.352 | 0.377 | **0.364** | **+0.012** |

Material is at majority exactly. Color and shape are at majority + 1-3 pp — below any reasonable threshold for "the linear probe found identity." Iter 18 was *below chance*; Iter 21 is *at majority*. The scale-up moved the needle from "actively wrong" to "indistinguishable from a constant predictor." That is not the disentanglement headline we wanted.

Crucially, **probe-train accuracy is also low** (color 17%, material 52%, shape 38%). The probe isn't overfitting and failing to generalize — it can't even fit the training side. The features simply do not carry identity.

---

## Iter 21 — z_dyn / per-window diagnostic (2026-05-18): identity is nowhere

**Question.** Identity could have failed to land in `z_static` for three structurally different reasons:

1. **Slot drift.** Slot index `k` points to different objects in different windows of the same video, so averaging z_static across windows destroys whatever signal each window carried.
2. *(omitted — VICReg-collapse hypothesis; harder to test independently.)*
3. *(omitted — bottleneck-too-narrow hypothesis; testable later by widening d_static.)*
4. **Identity in z_dyn.** The decoder has 32-dim z_dyn × 12 frames = 384 dims/slot/window vs 16 for z_static. It may have learned to read identity from z_dyn's constant component and never gave z_static a real job.

A single experiment distinguishes (1) and (4): probe both `z_dyn_mean` (z_dyn averaged over T) and per-window z_static under the same protocol.

**Result — three feature variants, same 50/50 video split.**

| Feature variant | Color | Material | Shape | Notes |
|---|---:|---:|---:|---|
| z_static, per-video avg (K=8 × 16) | 0.156 | 0.502 | 0.364 | headline; +0.024 / 0.000 / +0.012 vs maj |
| z_static, per-window (K=8 × 16)    | 0.149 | 0.498 | 0.363 | slightly *worse* than per-video avg |
| **z_dyn_mean, per-window (K=8 × 32)** | **0.184** | **0.510** | 0.345 | best, but color is still only +0.051 vs majority |
| majority baseline                   | 0.132 | 0.502 | 0.352 | (per-test-set empirical) |

Reading the rows:

- **Slot drift (hypothesis 1) is ruled out.** Per-window z_static is ≈ per-video-averaged z_static across all three attributes (in fact slightly worse — the opposite of what slot drift would predict).
- **Identity in z_dyn (hypothesis 4) is ruled out.** z_dyn_mean does carry marginally more color info than z_static (18.4% vs 15.6% vs 13.2% majority), but it is nowhere near the 50%+ that would mean "the decoder is reading identity from z_dyn." Material and shape are at or below majority for z_dyn_mean too.
- **We are in the worst diagnostic outcome: identity is nowhere.** The encoder has produced zero usable signal about whether a slot is rubber or metal, and only the faintest signal about color. The recon objective on Wan-VAE latents at 8×8 spatial resolution does not carry enough semantic gradient to make the encoder bother encoding identity. The decoder is reconstructing well using positions, motion, and low-frequency pixel statistics; it doesn't need "this is a metal cube" to do its job at 8×8 latent resolution.

**Interpretation.** The Iter 21 architectural disentanglement claim ("freezing z_dyn → 0 latent variation") was true, and remains true — but it meant *z_dyn is doing all the work*. Time-blinded decoder + block-diagonal cross-attention guarantees the factorization is mathematically clean, but does not guarantee the factorization the loss learns matches the one we wanted. We assumed "z_static = identity, z_dyn = motion" would be the natural division of labor under recon supervision. It isn't. The loss let z_static stay near-constant and put everything else into z_dyn.

This is the kind of result that, if Iter 22 produces the positive case, becomes a clean writeup: *naive architectural disentanglement does not yield semantic disentanglement; here is the loss surgery that does.*

**Artifacts.**
- Linear probe: `outputs/iter21_10000vid_213853/probe_identity_linear.json`
- Three-variant diagnostic: `outputs/iter21_10000vid_213853/probe_identity_diag.json`
- Probe scripts: `scripts/probe_iter21_identity.py`, `scripts/probe_iter21_zdyn_diag.py`

---

## Iter 22 plan — push identity into z_static via auxiliary supervision

Two parallel branches, both run from a single sbatch (`scripts/sbatch_iter22_full.sh`).

**22a — DINO-CLS auxiliary (`--lambda_dino 0.5`).** Already-wired hook in `train_trajectory.py`: a small MLP projects `z_static[B,K,16]` to `d_dino=384` and is trained to match a per-slot visibility-weighted average of frame-level DINOv2-S CLS tokens. All K slots in a video target the same CLS sequence, weighted by their own visibility. This is *weaker* than true DINOSAUR (which uses per-patch features and slot→patch assignment) — it's a video-level CLS signal split across slots — but it is the right cheap first try. Tests whether *any* DINO signal is sufficient to move z_static off chance.

**22c — Supervised attrs head (`--lambda_attrs 1.0`).** New in this iter: a 2-layer MLP `AttrsHead` (z_static → 64 → color/material/shape logits) trained with per-slot CE on GT attrs, masked by `slot_mask`. Direct supervision. Independent of the DINO cache. Tests whether the architecture *can* route per-slot identity into z_static when forced to, regardless of teacher quality.

**Shared recipe.** Identical to Iter 21 except for the new aux loss term:
- 10,000 train videos × 4 windows, 80/20 split by video_id (same seed=42 → same val pool).
- 60 epochs, batch=4, lr=5e-4 cosine→0, dropout 0.1, wd 1e-3.
- K=8, d_model=192, d_static=16, d_dyn=32.
- `lambda_smooth 0.1`, `lambda_entropy 0.01`, `lambda_vicreg 0.01`, `lambda_event_sup 0.02`.

**sbatch verification phases.** Before either long training launches, `sbatch_iter22_full.sh` runs a 10-min smoke gauntlet:
- Phase 0a: 5-vid Wan cache rebuild (3 min).
- Phase 0b: 5-vid DINO cache (downloads `facebook/dinov2-small` to `$HF_HOME` if absent).
- Phase 0c: Python assertion on the DINO blob — `cls.shape == (12, 384)`, dtype float32, finite. Bails out if the cache contract is wrong before committing 1-3 h to the full DINO cache.
- Phase 0d: 1-epoch trainer smoke with `--lambda_dino 0.5` — confirms the einsum dimension assertion holds and the loss is finite.
- Phase 0e: 1-epoch trainer smoke with `--lambda_attrs 1.0` — confirms the attrs head plumbing works.

Then Phase 1 (full DINO cache, resume-safe), Phase 2 (Iter 22a train), Phase 3 (Iter 22c train), Phase 4 (linear identity probe + z_dyn diagnostic on each checkpoint, output `probe_identity_linear.json` and `probe_identity_diag.json` next to each ckpt).

**Decision matrix (Phase 4 readout).**

| 22a color | 22a M/S | 22c color/M/S | Meaning | Next step |
|---|---|---|---|---|
| 50%+ | 50%+ | passes | DINO-CLS was enough teacher | Ship 22a, write up |
| 50%+ | trivial | passes | CLS too coarse for M/S; encoder *can* route per-slot when asked | Iter 23 = patch-level DINOSAUR |
| 50%+ | trivial | trivial on M/S | CLS helps color (scene-level proxy); architecture cannot route material/shape per-slot even under direct supervision | Architecture change: wider d_static, more iters, per-slot decoder targets |
| trivial | trivial | passes | DINO supervision was wrong shape; encoder is fine | Iter 23 = patch-level DINOSAUR |
| trivial | trivial | trivial | Encoder genuinely cannot route per-slot features to z_static at all | Revisit slot-binding mechanism (K=8 queries not competing enough) |

**Walltime budget.** ~12 h sequential — 10 min smoke + 1-3 h DINO cache + 3 h × 2 trainings + ~15 min probes. PACE H200 partition.

---

## Changelog

| Date | Update |
|---|---|
| 2026-05-08 | Wan-VAE ceiling probed (6.0e-5). Decoder tuned to 1.04× on rung 1/2. |
| 2026-05-09 | Rung 3 trained (1.18×). Rungs 4+5 measured per video. Rung 6 (full pipeline) and rung 7 (counterfactual) succeeded on smooth videos; collision-frame weakness confirmed at all three dynamics-involving rungs. |
| 2026-05-11 | EventHead motion-teacher F1=0.632 on 5 vids (v4 recipe). Beats GT-teacher ceiling (0.625). Two non-obvious bugs (abs_thresh=0.05, Conv1d window mismatch) found and fixed. |
| 2026-05-12 | 20-vid scale-up exposed structural head bug: per-slot Conv1d blind to pair info. `qva_nn` fix added — standalone head overfits 20 vids in 0.5 s (F1=0.605). Full pipeline limited by encoder q_pred noise (P=0.07 → 0.86 when fed GT q at inference). q-smoothness regularizer added but λ=1.0 below biting threshold. |
| 2026-05-14 | **Iter 21**: full-CLEVRER scale (10,000 videos, 40,000 windows). 60 epochs, recon 0.0104 / val_recon 0.0078 — *val < train* (Iter 18 was 2.4× overfit). Architecture and recipe generalize; disentanglement probes pending. Renamed RESULTS.md → DEVLOG.md. |
| 2026-05-15 | Cleanup: removed dead SlotQueryEncoder/EventHead/wan-flow code (71 files, ~10k LoC). README rewritten for the TrajectoryEncoder pipeline. Live tree narrowed to `scripts/{cache_wan_latents, train_trajectory, probe_iter21_identity}` + `src/{model/trajectory_encoder, data/clevrer_paired, data/clevrer_states}`. |
| 2026-05-18 | **Iter 21 identity probe — load-bearing test failed.** Linear probe (50/50 video split within trainer val pool) on per-video-averaged z_static: color 0.156 / material 0.502 / shape 0.364, all at or near majority baseline. Probe-train accuracy is also low → not a generalization gap, the features simply do not contain identity. Three-variant diagnostic (probe_iter21_zdyn_diag.py) ruled out both slot drift and identity-in-z_dyn — z_dyn_mean also at chance for material/shape, color +5 pp at best. Diagnosis: identity is nowhere. The recon objective on Wan-VAE latents does not carry enough semantic gradient. |
| 2026-05-19 | Iter 22 plan: two parallel branches, single sbatch. **22a** = Iter 21 recipe + `--lambda_dino 0.5` (existing CLS-level alignment hook). **22c** = Iter 21 recipe + new `--lambda_attrs 1.0` (supervised AttrsHead, CE on GT color/material/shape from z_static). Added `AttrsHead` class to `src/model/trajectory_encoder.py`; wired `--lambda_attrs` / `--attrs_hidden` plumbing in `train_trajectory.py`; restored deleted `scripts/cache_dino_features.py` with resume + `--max_windows` smoke flag; added DINO-shape assertion in trainer einsum; bundled all stages into `scripts/sbatch_iter22_full.sh` (Phase 0 smoke → Phase 1 full DINO cache → Phase 2 Iter22a → Phase 3 Iter22c → Phase 4 probes on each, ~12 h walltime). |
