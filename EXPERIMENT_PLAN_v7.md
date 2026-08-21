# DIALGA experiment plan — grounded in VideoFlexTok / FlexTok / PV-VAE

Date: 2026-07-16. Framing: **video tokenizer / embedding**, not a codec.

---

## RESOLVED 2026-07-16 — E1 ran (job 11199935). VERDICT: NOT A TOKENIZER PAPER.

The gate fired against us. Per the decision rule in §2/E1, we fall back to the
embedding-only claim and report the reconstruction result as the central negative.

| run | floats | kbps | rFVD | PSNR | LPIPS |
|---|---|---|---|---|---|
| wan_vae_ceiling | — | — | **1.86** | 48.31 | 0.0011 |
| **h264** | — | 34.3 | **41.75** | 43.25 | 0.0091 |
| h265 | — | 37.0 | 164.56 | 40.86 | 0.0212 |
| ours ep50 | 7684 | 186.3 | 226.58 | 33.14 | 0.0776 |
| ours ep50 | 3844 | 93.2 | 228.00 | 33.15 | 0.0780 |
| ours ep50 | 964 | 23.4 | 278.95 | 32.55 | 0.0926 |
| ours ep50 | 1924 | 46.6 | 281.67 | 32.75 | 0.0871 |
| ours ep50 | 644 | 15.6 | 335.33 | 32.45 | 0.1356 |

ep200: 964f → 153.04 · 1924f → **179.32 (worse)** · 3844f → 119.98 · h264 → 41.74

1. **H.264 beats us on rFVD too, 3–5×.** We now lose to a 2003 codec on PSNR, LPIPS AND
   rFVD. The friendlier metric did not rescue us — the problem was never the metric. We are
   blurry *and* distributionally wrong.
2. **There is no real rate axis.** rFVD moves with *training* (964f: 279→153) but with
   *rate* it is weak and non-monotonic (ep200: 964→1924 makes rFVD WORSE). Same saturation
   PSNR showed, now confirmed on the field's own headline metric. The model ignores its bits.
3. **Not close to published tokenizers.** VideoFlexTok rFVD 48.7 @160 tok on K600 128²;
   LARP 42.1; VidTok-FSQ 84.1. We are 120–335 on CLEVRER, a far easier dataset.

**E2 (flow decoder) is DROPPED.** Flow buys sharpness — ~2× rFVD in FlexTok's own Table 3.
We need 3× just to reach H.264 on easy data, before approaching any tokenizer baseline.
Not worth ~a month of compute at 2 h/epoch. **E4 (nested rate) dropped with it** — nesting a
rate axis that does not exist is pointless.

**THE PAPER = the embedding result** (job 11199805, §1): 96 learned floats beat the full
27648-float raw Wan latent (+0.16 colour mAP, 288× smaller), the dim-matched PCA-96 control
(+0.25), and the untrained architecture (+0.18). DINOv2 0.950 vs ours 0.932 is the honest
caveat (it is our own MAE teacher). Reconstruction becomes a probe; the codec/rFVD tables
become a limitations section: **our latents are semantic, not compressive.**

Still live: **E3** (probe vs rate curve, zero new training) and **E5** (second dataset,
now the main reviewer risk since CLEVRER-only + a negative recon result is thin).

---

## 0. What the reference papers actually do (verified)

| | PV-VAE (2605.02134) | VideoFlexTok (2604.12887) | FlexTok (2502.13967) |
|---|---|---|---|
| Headline metric | FVD / rFVD | **rFVD, gFVD**, Cls.Score, ViCLIP | **rFID**, gFID, DreamSim |
| PSNR reported? | yes, secondary (32.26 K400) | **never** | appendix Table 8 only |
| H.264/H.265 baseline? | **no** | **no** | **no** (cites JPEG in related work only) |
| Rate–distortion curve? | no (fixed `t4s16c64`) | yes, gFVD vs #tokens (5–1280) | yes, Fig.5 rFID vs K∈{1..256} + bytes col |
| Repr. eval | frozen LVDM layer-14 feats → flow/track/next-frame | qualitative only (Fig.4) | **linear probe on frozen tokens, App. B** |

### The three findings that reset our strategy

1. **PSNR is not the target.** FlexTok's best model, full 256-token rate (512 bytes/img):
   rFID **1.08**, PSNR **17.70 dB**, LPIPS 0.219. They ship that happily. A rectified-flow
   decoder hallucinates sharp detail — tanks PSNR, wins FID/FVD. Our deterministic decoder
   does the exact opposite.
2. **Nobody in this line compares to H.264.** Our `rd_table` H.264 result (they beat us by
   10 dB at matched bitrate) is real but is *not* the axis this field is judged on.
   → demote to an honest limitations paragraph. Do **not** headline it.
3. **PV-VAE IS MIS-CITED IN OUR NOTES.** It has **no static/dynamic factorization** — it is a
   single unstructured spatial grid (`t4s16c64`), and its novelty is a *training objective*
   (drop future frames; encoder sees only past). The correct cites for factorized
   static/dynamic-with-spatial-grid are **VidTwin (Structure+Dynamics)** and **PVDM**.
   `memory/project_v57_spatial_grid.md` must be corrected.

### The comparison we must NOT make
FlexTok's 17.70 dB is ImageNet 256². Ours is CLEVRER 128², static camera, constant
background, ~5 rigid objects. **"We beat FlexTok by 15 dB" is not a claim.** Only the
methodological point transfers.

---

## 1. Where we actually stand

| Evidence | Status |
|---|---|
| PSNR/LPIPS/SSIM + codec R–D curve, 5 rates | DONE (`rd_table`, job 11199622) |
| Frozen set-probe vs base-rate prior | DONE (job 11173582): colour mAP 0.932 @ep200, monotone |
| Frozen probe vs *real* baselines (PCA/random/DINO/raw) | RUNNING (job 11199805) |
| **rFVD — the headline metric** | **NEVER MEASURED** ← the gate |
| Nested/flexible rate (one model, many budgets) | **not built** — our sweep retrains per rate |
| Second dataset | not started |

**Our R–D curve is flat in PSNR** (4× bits → +0.9 dB). Under FlexTok's framing that is not
automatically a defect — their rate axis moves because rFID is distributional and a flow
decoder can act on it. But we cannot know which it is until rFVD is measured. If rFVD is
*also* flat, the model genuinely ignores its bits and no framing saves it.

---

## 2. Experiments, in dependency order

### E1 — rFVD table  ← **THE GATE. Launch first.**
`scripts/probes/rfvd_table.py`. cd-fvd (I3D), verified: identical→0.0, same-process→2.93,
vs-noise→3246. FlexTok Table-8 layout: `floats | bytes | rFVD | PSNR | SSIM | LPIPS`,
plus Wan-VAE ceiling and H.264/H.265 rows. 512 val videos, one 33-frame chunk each.
Rates: 644 / 964 / 1924 / 3844 / 7684, epoch-matched ep50 + converged ep200.

**Decision rule:**
- rFVD *varies* with rate → we have a rate axis; the flat PSNR curve was a metric artifact.
  Proceed to E2, headline rFVD.
- rFVD *flat too* → the model ignores its bits. No tokenizer paper. Fall back to the
  embedding-only claim (E3) and report this as the central negative result.

### E2 — Flow decoder head-to-head  (conditional on E1)
The single highest-leverage architectural change, and the one the literature says decides
everything: swap the deterministic decoder for a rectified-flow decoder at **identical
rate and identical frozen encoder**. This is exactly FlexTok's Table 3 ablation
(2D-MSE → 2D-flow → 1D-flow → +REPA: rFID 51.27 → 32.85 → 23.28 → 5.98). Our
`src/model/latent_decoder.py` flow decoder exists but is a toy; `v6_flow_clevrer.sbatch`
is HELD for that reason. Needs a real build before launch.
Cost warning: ~2 h/epoch at `--batch_size 2`. Raise batch first (see §3).

### E3 — Frozen-token linear probe vs budget  (FlexTok Appendix B protocol)
Extends the running `baseline_probe_table.py`. Their protocol: linear classifier on frozen
tokens, concat to one vector, sweep vs token count, compare against TiTok (which *degrades*
with budget because it retrains per budget, while nested FlexTok *improves*).
**Our PCA-96 control is stronger than their TiTok comparison** — keep it.
We already have the per-rate arms to plot mAP vs rate. This is a real figure with zero new
training. NOTE: our sweep is per-budget retraining = the TiTok arm, not the FlexTok arm.

### E4 — Nested / flexible rate  (the actual VideoFlexTok contribution)
Their headline is one model serving 5–1280 tokens via nested dropout. We retrain per rate.
Replicating this = training with nested dropout over z_dyn/z_static dims. New capability,
new training run. Only worth it if E1 and E2 land.

### E5 — Second dataset
CLEVRER-only will draw fire. Cheapest real option is the existing LIBERO Wan-latent cache
(`libero90_*`). Deprioritized until E1/E2 resolve.

---

## 3. The throughput blocker (read before any new training)
2 h/epoch, ~1.5 epochs/day on embers ⇒ any new arm is ~a month to ep50. Cause is
`--batch_size 2` on a 48 GB L40S — almost certainly 4–8× below what fits. Raising it breaks
step-comparability with the existing ep50/ep200 fleet, so it is only correct for a *fresh*
fleet (E2/E4). **Benchmark max batch before launching E2.**

---

## 4. What NOT to do
- Do **not** launch the 322/164-float ultra-low-rate arms. They would spend a month adding
  points to a curve that is flat and, in PSNR, already loses to H.264.
- Do **not** headline the codec comparison.
- Do **not** resume v57 (cancelled; was losing to mean-pool on the valid probe).
- Do **not** claim any cross-dataset PSNR win over FlexTok.
