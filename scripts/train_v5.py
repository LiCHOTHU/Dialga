"""scripts/train_v5.py — v5.1 chunk-wise trainer with stage-gated losses.

Stages (each is a contiguous range of epochs):

  Stage 1: L_recon only.                  Warm up the autoencoder.
  Stage 2: + L_pred + L_fwd + L_infonce.  Predictive pressure + contrastive
                                          identity pressure on z_static.
  Stage 3: + L_event_aux + L_gate.        Adds the supervised event channel.

Each training sample is a triplet of chunks from one video:
  chunk_obs    — encoded as the "observed" half
  chunk_pred   — predicted by rolling z_dyn forward via ForwardDynamics
  chunk_obs_b  — InfoNCE positive (different chunk, same video)
plus a scalar gate_GT (collision occurred in chunk_pred).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.data.droid_window import DroidChunkPairs
from src.loss.feature_pred import masked_feature_loss
from src.loss.info_nce import info_nce
from src.model.attrs_head import AttrsHead
from src.model.aux_semantic_decoder import AuxSemanticDecoder
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import (LatentDecoder, SpatialBroadcastDecoder,
                                       SlotDecoder, SpatialGridDecoder,
                                       FlowMatchingDecoder, FlowMatcher)
from src.model.latent_encoder import LatentEncoder3D
from src.model.camera_pose import CameraConditioner, synthetic_pan
from src.loss.vicreg import vicreg_var_cov
from src.model.masking import ObjectRegionMask


def load_wan_vae_trainable(model_id, dtype, device, train_enc: bool, train_dec: bool):
    """Load Wan-VAE and set requires_grad on encoder/decoder submodules per flags.
    Returns the VAE in train() mode for the unfrozen parts."""
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    # Default: freeze everything.
    for p in vae.parameters():
        p.requires_grad_(False)
    # Unfreeze the requested halves.
    if train_enc and hasattr(vae, "encoder"):
        for p in vae.encoder.parameters():
            p.requires_grad_(True)
    if train_dec and hasattr(vae, "decoder"):
        for p in vae.decoder.parameters():
            p.requires_grad_(True)
    vae.eval()  # keep BN/etc. in eval; only weights have grad
    return vae.to(device)


def _vae_encode(vae, pix: torch.Tensor) -> torch.Tensor:
    """pix (B, T_pix, 3, H, W) in [-1, 1] -> latent (B, C, T_lat, H_lat, W_lat).

    Wan-VAE expects (B, 3, T_pix, H, W). Matches scripts/cache_wan_latents.py."""
    x = pix.permute(0, 2, 1, 3, 4).contiguous().to(next(vae.parameters()).dtype)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.float()                              # (B, C, T_lat, H_lat, W_lat)


def _vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """latent (B, C, T_lat, H_lat, W_lat) -> pix (B, T_pix, 3, H, W) in [-1, 1]."""
    z = latent.to(next(vae.parameters()).dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out          # (B, 3, T_pix, H, W)
    return pix.permute(0, 2, 1, 3, 4).contiguous().float()

N_COLOR    = len(COLOR_VOCAB)
N_MATERIAL = len(MATERIAL_VOCAB)
N_SHAPE    = len(SHAPE_VOCAB)


def _modal_labels(attrs: torch.Tensor, slot_mask: torch.Tensor,
                  lo: int, hi: int) -> torch.Tensor:
    """Modal class index per video for attribute slice [lo:hi).

    attrs     : (B, K, A) one-hot blocks
    slot_mask : (B, K)    bool
    Returns   : (B,) long; -100 (CE ignore_index) where no real slots.
    """
    block = attrs[..., lo:hi]                              # (B, K, C)
    cls = block.argmax(dim=-1)                             # (B, K)
    n_class = hi - lo
    onehot = F.one_hot(cls, n_class).float()               # (B, K, C)
    masked = onehot * slot_mask.float().unsqueeze(-1)
    counts = masked.sum(dim=1)                             # (B, C)
    labels = counts.argmax(dim=-1)                         # (B,)
    valid = slot_mask.bool().any(dim=-1)                   # (B,)
    return torch.where(valid, labels, torch.full_like(labels, -100))


def stage_at_epoch(ep: int, s1: int, s2: int) -> int:
    """Two-stage training only. Stage 3 (events) was removed after the
    2026-05-20 120-epoch run showed it actively hurt val_recon (0.019 -> 0.021)
    when L_event_aux + L_gate gradients backed into the encoder. EventHead /
    GEvent / GatePredictor still instantiate but receive no gradient.
    """
    if ep <= s1: return 1
    return 2


def anneal_lambdas(ep, history, args, recon_target, pixel_target):
    """Loss-balance curriculum for --anneal_pixel.

    Phase A (start of training): latent only. lambda_recon = --anneal_recon_hi,
    lambda_pixel = --anneal_pixel_lo (~0), so L_recon is supervised purely in
    Wan-latent space and the model first nails latent fidelity.

    Trigger: when the *train* latent L_recon goes flat — its per-epoch relative
    improvement stays below --anneal_plateau_delta for --anneal_plateau_patience
    consecutive epochs (never before stage 1 ends).

    Phase B (after the trigger): linearly ramp over --anneal_ramp_epochs to the
    configured pixel-dominant targets (lambda_recon -> recon_target=args value,
    lambda_pixel -> pixel_target), so the pixel loss becomes dominant.

    Stateless by design: the trigger epoch is recomputed from the train-recon
    series in `history` on every call, so it is identical before and after a
    self-chain resume (history is restored from the checkpoint).

    Returns (lambda_recon, lambda_pixel, triggered: bool, trigger_ep|None).
    """
    hi_recon = float(args.anneal_recon_hi)
    lo_pixel = float(args.anneal_pixel_lo)
    series = sorted((h["epoch"], h["train"]["recon"]) for h in history
                    if h.get("train") and "recon" in h["train"] and h["epoch"] < ep)
    trigger_ep = None
    best = float("inf"); stale = 0
    min_ep = args.stage1_epochs + 1          # never trigger during stage-1 warmup
    for e, r in series:
        if r < best * (1.0 - args.anneal_plateau_delta):
            best = r; stale = 0
        else:
            stale += 1
        if e >= min_ep and stale >= args.anneal_plateau_patience:
            trigger_ep = e
            break
    if trigger_ep is None:
        return hi_recon, lo_pixel, False, None
    frac = min(1.0, max(0.0, (ep - trigger_ep) / max(1, args.anneal_ramp_epochs)))
    cur_recon = hi_recon + (recon_target - hi_recon) * frac
    cur_pixel = lo_pixel + (pixel_target - lo_pixel) * frac
    return cur_recon, cur_pixel, True, trigger_ep


# --------------------------------------------------------------------- losses

def compute_losses(batch, models, args, stage: int, device, vae=None):
    """Compute ALL six losses every step (logged regardless of stage).
    Stage-gating controls only which losses sum into `total`.

    Wiring (per v5.1 spec):
      * One encoder pass on each of chunk_obs / chunk_pred / chunk_obs_b.
      * z_static_a reused for both recon_obs and recon_pred (identity-reuse).
      * Event correction is injected into the rollout from stage 2 onward;
        GEvent is zero-init so stage 2 is naturally a no-op until L_event_aux
        starts moving g_event in stage 3.
      * L_fwd uses the BASE step (no event correction).
      * L_event_aux trains only event_head + g_event (both fwd and target
        encoded z_dyn are detached).

    v5.1.2 VAE-unfreeze additions (Exp 2/3/4):
      * If args.unfreeze_vae_enc and vae provided: re-encode pix_* through
        vae.encoder (with grad) and use those fresh latents as chunk_*.
      * If args.unfreeze_vae_dec and vae provided: decode recon_obs/recon_pred
        through vae.decoder (with grad), add L_pixel = MSE(pred_pix, pix_*).
    """
    enc, dec, fwd, eh, ge, gp, ah = models[:7]
    # extras (masker/auxd for MAE, cc for camera pose) are identified by type so
    # their order / presence is independent (cc is appended after masker/auxd).
    _extra = models[7:]
    masker = next((m for m in _extra if isinstance(m, ObjectRegionMask)), None)
    auxd   = next((m for m in _extra if isinstance(m, AuxSemanticDecoder)), None)
    cc     = next((m for m in _extra if isinstance(m, CameraConditioner)), None)
    use_pixels = getattr(args, "use_pixels", False)

    chunk_obs   = batch["chunk_obs"].to(device)       # (B, C, T, H, W)
    chunk_pred  = batch["chunk_pred"].to(device)
    chunk_obs_b = batch["chunk_obs_b"].to(device)
    gate_GT     = batch["gate_GT"].to(device).float() # (B,) in {0, 1}
    attrs       = batch["attrs"].to(device)            # (B, K, A) one-hot
    slot_mask   = batch["slot_mask"].to(device)        # (B, K) bool

    # v5.1.2: re-encode raw pixels through Wan-VAE encoder (with grad) so it
    # gets a gradient signal. Cached latents are overridden.
    if vae is not None and getattr(args, "unfreeze_vae_enc", False):
        pix_obs   = batch["pix_obs"].to(device)        # (B, T_pix, 3, H, W)
        pix_pred  = batch["pix_pred"].to(device)
        pix_obs_b = batch["pix_obs_b"].to(device)
        chunk_obs   = _vae_encode(vae, pix_obs)
        chunk_pred  = _vae_encode(vae, pix_pred)
        chunk_obs_b = _vae_encode(vae, pix_obs_b)

    # ---- v5.8 camera pose: warp with a KNOWN synthetic camera, condition on it ----
    # Reference (clean chunk, identity pose) is the invariance TARGET: from the
    # warped view + its pose the encoder must reproduce the clean-view code, i.e.
    # invert the known camera. Blind control feeds zero pose (camera withheld).
    pemb_o = pemb_p = pemb_b = None
    z_dyn_ref = z_static_ref = None
    real_cam = False
    if cc is not None and getattr(args, "cam_source", "synth") == "real":
        # DROID: the KNOWN per-chunk camera trajectory rides in the batch. Motion
        # is already in the real video (no synthetic warp) and there is no
        # un-warped canonical reference. Instead we use a WITHIN-EPISODE
        # persistence pair: chunk_obs and chunk_obs_b are two chunks of the SAME
        # episode -> SAME static scene under DIFFERENT cameras. We relativise
        # chunk B's ABSOLUTE pose to chunk A's frame 0 so both canonical static
        # grids land in ONE common episode reference; L_cam_inv (below) then
        # requires their pose-registered grids to agree. This makes the known
        # camera CAUSALLY earn its keep on real data (see L_cam_inv block).
        real_cam = True
        pose_o = batch["pose_obs"].to(device)             # (B, T_lat, pose_dim) absolute
        pose_p = batch["pose_pred"].to(device)
        pose_b = batch["pose_b"].to(device)
        anchor = pose_o[:, :1, :]                          # chunk A frame0 = common ref
        pemb_o = cc.embed(cc.relative(pose_o))             # A rel A-frame0 (decoder re-applies)
        pemb_p = cc.embed(cc.relative(pose_p))             # next chunk rel its own frame0
        pemb_b = cc.embed(cc.relative_to(pose_b, anchor))  # B rel A-frame0 -> COMMON frame
        if getattr(args, "cam_pose_blind", False):
            pemb_o, pemb_p, pemb_b = (torch.zeros_like(pemb_o),
                                      torch.zeros_like(pemb_p),
                                      torch.zeros_like(pemb_b))
    elif cc is not None:
        Bc, Tc = chunk_obs.shape[0], chunk_obs.shape[2]
        pemb_id = cc.embed(cc.relative(torch.zeros(Bc, Tc, cc.pose_dim, device=device)))
        if getattr(args, "cam_pose_blind", False):
            pemb_id = torch.zeros_like(pemb_id)
        with torch.no_grad():
            ref = enc(chunk_obs, pose_emb=pemb_id)
            z_dyn_ref, z_static_ref = ref["z_dyn"], ref["z_static"]
        ms, mz = args.cam_max_shift, args.cam_max_zoom
        chunk_obs,   pose_o = synthetic_pan(chunk_obs,   ms, mz)
        chunk_pred,  pose_p = synthetic_pan(chunk_pred,  ms, mz)
        chunk_obs_b, pose_b = synthetic_pan(chunk_obs_b, ms, mz)
        pemb_o = cc.embed(cc.relative(pose_o))
        pemb_p = cc.embed(cc.relative(pose_p))
        pemb_b = cc.embed(cc.relative(pose_b))
        if getattr(args, "cam_pose_blind", False):
            pemb_o, pemb_p, pemb_b = (torch.zeros_like(pemb_o),
                                      torch.zeros_like(pemb_p),
                                      torch.zeros_like(pemb_b))

    # ---- three encoder passes ----
    enc_obs  = enc(chunk_obs,   pose_emb=pemb_o)
    enc_pred = enc(chunk_pred,  pose_emb=pemb_p)
    enc_b    = enc(chunk_obs_b, pose_emb=pemb_b)
    z_static_a = enc_obs["z_static"]                  # (B, D_s)
    z_dyn_obs  = enc_obs["z_dyn"]                     # (B, T, D_d)  — per-frame
    z_dyn_pred_target = enc_pred["z_dyn"].detach()    # (B, T, D_d), detached
    z_static_b = enc_b["z_static"]

    # ---- last observed frame's z_dyn drives rollout/event prediction ----
    z_dyn_last = z_dyn_obs[:, -1]                     # (B, D_d)
    T_chunk = z_dyn_obs.shape[1]

    # ---- rollout: ONE chunk-to-chunk step, expands to T per-frame outputs ----
    z_dyn_pred_base = fwd.chunk_step(z_dyn_last, T_chunk)     # (B, T, D_d)
    z_event = eh(z_dyn_last)                                  # (B, D_e)
    correction = ge(z_event)                                  # (B, D_d) — zero at init
    # broadcast correction to all predicted frames, scaled by gate
    correction_t = correction.unsqueeze(1).expand(-1, T_chunk, -1)
    z_dyn_pred_roll = z_dyn_pred_base + correction_t * gate_GT.unsqueeze(-1).unsqueeze(-1)

    # ---- reconstructions (z_static_a reused for both) ----
    # v5.3: a SlotDecoder consumes the per-slot code z_slots (B, M, D_s) instead
    # of the pooled global z_static; everything else (InfoNCE / attrs / MAE /
    # diagnostics) keeps using the pooled z_static_a so the rest of the stack is
    # unchanged.
    # v5.7: a SpatialGridDecoder consumes the spatial static grid
    # z_static_grid (B, c, g, g) instead of the flattened vector; InfoNCE /
    # attrs / MAE / diagnostics keep using the flat z_static_a.
    is_flow = isinstance(dec, FlowMatchingDecoder)
    if isinstance(dec, SlotDecoder):
        dec_cond_a = enc_obs["z_slots"]
    elif isinstance(dec, (SpatialGridDecoder, FlowMatchingDecoder)):
        dec_cond_a = enc_obs["z_static_grid"]
    else:
        dec_cond_a = z_static_a

    # ---- reconstruction objective ----
    # v6 flow decoder: the decoder is a rectified-flow velocity net, so the
    # "reconstruction" loss is the MinRF velocity MSE on the Wan latent
    # (target v = eps - x0), NOT a direct latent MSE. Sharp pixels come only at
    # sampling time -> no VAE decode in the loop. L_pred keeps the forward-
    # dynamics novelty by flowing the NEXT chunk from the rolled-out z_dyn.
    if is_flow:
        x_sig_o, sig_o, v_tgt_o = FlowMatcher.add_noise(chunk_obs)
        L_recon = F.mse_loss(dec(x_sig_o, sig_o, dec_cond_a, z_dyn_obs), v_tgt_o)
        x_sig_p, sig_p, v_tgt_p = FlowMatcher.add_noise(chunk_pred)
        L_pred = F.mse_loss(dec(x_sig_p, sig_p, dec_cond_a, z_dyn_pred_roll), v_tgt_p)
        recon_obs = recon_pred = None
    else:
        recon_obs = dec(dec_cond_a, z_dyn_obs, pose_emb=pemb_o)
        recon_pred = dec(dec_cond_a, z_dyn_pred_roll, pose_emb=pemb_p)
        L_recon = F.mse_loss(recon_obs, chunk_obs)   # chunk_obs is warped when cc on
        L_pred = F.mse_loss(recon_pred, chunk_pred)
    L_fwd = F.mse_loss(z_dyn_pred_base, z_dyn_pred_target)

    # v5.1.2: pixel-space supervision. Decode the reconstructed latents through
    # the Wan decoder and MSE against raw pixels. Done OUTSIDE no-grad so grad
    # flows through the decoder (frozen OR trainable) back to recon_*, the small
    # model, and — when the encoder is unfrozen — all the way to the Wan encoder.
    # This is the ANCHOR that keeps an unfrozen encoder's latents decodable:
    # without it (Exp 2/e2) the encoder sits on both sides of the latent-space
    # L_recon and collapses to a trivial low-variance blob (recon -> 0, pixels ->
    # garbage). Active whenever a VAE is loaded and lambda_pixel > 0, independent
    # of which half is unfrozen.
    L_pixel = torch.zeros((), device=device)
    L_pixel_pred = torch.zeros((), device=device)
    pixel_on = (vae is not None and getattr(args, "lambda_pixel", 0.0) > 0.0
                and "pix_obs" in batch and not is_flow)
    if pixel_on:
        pix_obs_target  = batch["pix_obs"].to(device)
        pix_pred_target = batch["pix_pred"].to(device)
        pred_pix_obs   = _vae_decode(vae, recon_obs)        # (B, T_pix, 3, H, W)
        pred_pix_pred  = _vae_decode(vae, recon_pred)
        L_pixel      = F.mse_loss(pred_pix_obs,  pix_obs_target)
        L_pixel_pred = F.mse_loss(pred_pix_pred, pix_pred_target)
    # Ablation 1: --consist_loss mse replaces InfoNCE with plain MSE on z_static.
    if getattr(args, "consist_loss", "infonce") == "mse":
        L_infonce = F.mse_loss(z_static_a, z_static_b)
    else:
        L_infonce = info_nce(z_static_a, z_static_b,
                             temperature=args.infonce_temperature)

    # L_event_aux: at gate=1, g_event(event_head(z_dyn_last)) (broadcast across T)
    # must match (z_dyn_pred_target - z_dyn_pred_base). Both sides on the target
    # detached so EventHead+GEvent are the only modules updated by this term.
    true_residual = z_dyn_pred_target - z_dyn_pred_base.detach()
    pred_residual_t = ge(eh(z_dyn_last)).unsqueeze(1).expand(-1, T_chunk, -1)
    L_event_aux = (F.mse_loss(pred_residual_t, true_residual, reduction="none")
                   * gate_GT.unsqueeze(-1).unsqueeze(-1)).mean()

    gate_logits = gp(z_dyn_last)
    L_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_GT)

    # ---- L_mae_sem (v5.1.3 / MAETok Fork-C): masked semantic prediction ----
    # A SECOND encoder pass on a saliency-masked copy of chunk_obs; the aux
    # decoder must predict frozen DINOv2 patch features at the masked cells
    # from (z_static_m, z_dyn_m) alone. HARD INVARIANTS:
    #   * ForwardDynamics / recon / InfoNCE consume only the UNMASKED pass;
    #   * the masked pass feeds ONLY this loss (its codes are discarded after).
    L_mae_sem = torch.zeros((), device=device)
    # Trivial floor: 1 - cos(mean masked target, each masked target). If L_mae_sem
    # is not well below this, the aux head is just predicting the mean feature and
    # the masking is NOT forcing the code to carry structure (Reason 2 / saturation
    # check for the gaze-MAE smoke).
    mae_base = torch.zeros((), device=device)
    mae_on = (masker is not None and auxd is not None
              and getattr(args, "lambda_mae", 0.0) > 0.0 and "dino_obs" in batch)
    if mae_on:
        chunk_masked, cell_mask = masker(chunk_obs.detach())
        enc_m = enc(chunk_masked)
        dino_pred = auxd(enc_m["z_static"], enc_m["z_dyn"])   # (B, D_feat, T, H, W)
        dino_tgt = batch["dino_obs"].to(device)               # (B, T, H, W, D_feat)
        L_mae_sem = masked_feature_loss(dino_pred, dino_tgt, cell_mask)
        with torch.no_grad():
            if cell_mask.any():
                tm = dino_tgt[cell_mask].float()              # (M, D)
                mean_feat = tm.mean(dim=0, keepdim=True).expand_as(tm)
                mae_base = (1.0 - F.cosine_similarity(mean_feat, tm, dim=-1)).mean()

    # ---- L_attrs (Exp 1): supervised CE on z_static for color/material/shape ----
    # Modal label per video (most-common class across visible slots); CE
    # ignores videos with no real slots via ignore_index=-100. Guarded by
    # lambda_attrs>0 so datasets without attribute labels (DROID) skip it
    # cleanly (empty attrs -> all-ignored CE would be NaN).
    L_attrs = torch.zeros((), device=device)
    if getattr(args, "lambda_attrs", 0.0) > 0:
        attrs_logits = ah(z_static_a)
        y_color    = _modal_labels(attrs, slot_mask, 0,                          N_COLOR)
        y_material = _modal_labels(attrs, slot_mask, N_COLOR,                    N_COLOR + N_MATERIAL)
        y_shape    = _modal_labels(attrs, slot_mask, N_COLOR + N_MATERIAL,       N_COLOR + N_MATERIAL + N_SHAPE)
        L_attrs = (F.cross_entropy(attrs_logits["color"],    y_color,    ignore_index=-100)
                   + F.cross_entropy(attrs_logits["material"], y_material, ignore_index=-100)
                   + F.cross_entropy(attrs_logits["shape"],    y_shape,    ignore_index=-100))

    # ---- L_vic (v5.2): VICReg variance+covariance on z_static & z_dyn ----
    # Forces per-dim variance up and decorrelates dims, raising the latent's
    # participation ratio (inspection: ~110/964 floats effective). Applied to
    # the observed-chunk codes; z_dyn flattened over (B, T).
    L_vic_var = torch.zeros((), device=device)
    L_vic_cov = torch.zeros((), device=device)
    vic_on = (getattr(args, "lambda_vic_var", 0.0) > 0.0
              or getattr(args, "lambda_vic_cov", 0.0) > 0.0)
    if vic_on:
        gamma = getattr(args, "vic_gamma", 1.0)
        vs_var, vs_cov = vicreg_var_cov(z_static_a, gamma=gamma)
        vd_var, vd_cov = vicreg_var_cov(z_dyn_obs.reshape(-1, z_dyn_obs.shape[-1]),
                                        gamma=gamma)
        L_vic_var = vs_var + vd_var
        L_vic_cov = vs_cov + vd_cov

    # ---- L_indep (v6.1): cross-code independence -> push identity OUT of z_dyn ----
    # Penalizes cross-correlation between z_static and the time-pooled z_dyn. InfoNCE
    # already routes identity into z_static; this actively decorrelates z_dyn from it,
    # fixing the residual identity leakage that a shared-trunk baseline also shows.
    L_indep = torch.zeros((), device=device)
    if getattr(args, "lambda_indep", 0.0) > 0.0:
        from src.loss.vicreg import cross_decorr
        L_indep = cross_decorr(z_static_a, z_dyn_obs.mean(dim=1))

    # ---- L_cam_inv (v5.8): the code from the warped view + known pose must match
    # the code from the clean view -> the encoder INVERTED the known camera. ----
    L_cam_inv = torch.zeros((), device=device)
    if cc is not None and z_dyn_ref is not None:
        # synth path: code from the warped view + known pose must match the
        # clean-view code -> the encoder INVERTED the known camera.
        L_cam_inv = (F.mse_loss(z_dyn_obs, z_dyn_ref)
                     + F.mse_loss(z_static_a, z_static_ref))
    elif real_cam:
        # real DROID PERSISTENCE: chunk A and chunk B are two chunks of the SAME
        # episode -> same static scene, different camera. Their pose-registered
        # canonical static codes must agree (pemb_b was relativised to A's frame0
        # so both grids live in one common frame). ON can register B's grid into
        # A's frame using the known pose; BLIND (zero pose) cannot without washing
        # out the spatial content L_recon needs -> this term is what makes the
        # known camera causally earn its keep on real data. Operate on the SPATIAL
        # grid (the pool has no axis for pose to act on); unit-normalise so it
        # can't be gamed by inflating magnitude (task #94); stop-grad the target
        # (BYOL-style) and lean on L_recon + InfoNCE to prevent grid collapse.
        ga_grid, gb_grid = enc_obs.get("z_static_grid"), enc_b.get("z_static_grid")
        if ga_grid is not None and gb_grid is not None:
            ga = F.normalize(ga_grid.flatten(1), dim=-1)
            gb = F.normalize(gb_grid.flatten(1), dim=-1).detach()
        else:                                            # mean/slot pool fallback
            ga = F.normalize(z_static_a, dim=-1)
            gb = F.normalize(z_static_b, dim=-1).detach()
        L_cam_inv = F.mse_loss(ga, gb)

    # ---- stage gating: which contribute to total ----
    # Stage 3 (events) was removed after it regressed val_recon. L_event_aux and
    # L_gate are still computed (for logging) but never enter `total`. EventHead /
    # GEvent / GatePredictor accordingly receive no gradient and stay near init.
    # L_attrs joins from stage 2 with weight λ_attrs (Exp 1: 0.05).
    lambda_recon = getattr(args, "lambda_recon", 1.0)
    cam_on = cc is not None
    if stage == 1:
        total = lambda_recon * L_recon
        if pixel_on:
            total = total + args.lambda_pixel * L_pixel
        if cam_on:
            total = total + args.lambda_cam_inv * L_cam_inv
    else:  # stage 2 — production loss for the rest of training
        total = (lambda_recon * L_recon
                 + args.lambda_pred * L_pred
                 + args.lambda_fwd * L_fwd
                 + args.lambda_consist * L_infonce
                 + args.lambda_attrs * L_attrs)
        if cam_on:
            total = total + args.lambda_cam_inv * L_cam_inv
        if pixel_on:
            total = total + args.lambda_pixel * (L_pixel + L_pixel_pred)
        if mae_on:
            total = total + args.lambda_mae * L_mae_sem
        if vic_on:
            total = (total + getattr(args, "lambda_vic_var", 0.0) * L_vic_var
                     + getattr(args, "lambda_vic_cov", 0.0) * L_vic_cov)
        if getattr(args, "lambda_indep", 0.0) > 0.0:
            total = total + args.lambda_indep * L_indep

    # ---- diagnostics (always logged, never contribute to loss) ----
    with torch.no_grad():
        diag = {
            "z_static_norm":  z_static_a.norm(dim=-1).mean(),
            "z_static_std":   z_static_a.std(dim=0).mean(),
            "z_dyn_obs_norm": z_dyn_obs.norm(dim=-1).mean(),    # avg over (B,T)
            "z_dyn_obs_std":  z_dyn_obs.std(dim=0).mean(),
            "z_dyn_roll_norm": z_dyn_pred_roll.norm(dim=-1).mean(),
            "z_event_norm":   z_event.norm(dim=-1).mean(),
            "gate_GT_mean":   gate_GT.mean(),
        }

    return {
        "recon": L_recon, "pred": L_pred, "fwd": L_fwd, "consist": L_infonce,
        "event_aux": L_event_aux, "gate": L_gate, "attrs": L_attrs,
        "pixel": L_pixel, "pixel_pred": L_pixel_pred, "mae_sem": L_mae_sem,
        "mae_base": mae_base,
        "vic_var": L_vic_var, "vic_cov": L_vic_cov, "indep": L_indep,
        "cam_inv": L_cam_inv,
        "total": total,
        **diag,
    }


# ----------------------------------------------------------------- validation

@torch.no_grad()
def validate(models, val_loader, args, device, vae=None):
    [m.eval() for m in models]
    sums = {}
    n = 0
    for batch in val_loader:
        # stage=3 so every loss + every diagnostic is computed
        out = compute_losses(batch, models, args, stage=3, device=device, vae=vae)
        B = batch["chunk_obs"].shape[0]
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v) * B
        n += B
    [m.train() for m in models]
    return {k: v / max(n, 1) for k, v in sums.items()}


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--out_dir",   type=str, required=True)
    ap.add_argument("--epochs",        type=int, default=24)
    ap.add_argument("--stage1_epochs", type=int, default=3)
    ap.add_argument("--stage2_epochs", type=int, default=16)
    ap.add_argument("--batch_size",    type=int, default=16)
    ap.add_argument("--num_workers",   type=int, default=0)
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-3)
    ap.add_argument("--lr_schedule",   type=str,   default="cosine",
                    choices=["cosine", "constant"])

    ap.add_argument("--d_static", type=int, default=32)
    ap.add_argument("--d_dyn",    type=int, default=16)
    ap.add_argument("--d_state",  type=int, default=8)
    ap.add_argument("--d_event",  type=int, default=4)
    ap.add_argument("--enc_hidden_ch", type=int, default=32)
    ap.add_argument("--dec_hidden_ch", type=int, default=64)
    ap.add_argument("--chunk_size_lat", type=int, default=9)
    # v5.2: attention pooling + spatial-broadcast decoder + VICReg.
    ap.add_argument("--pool_type", type=str, default="mean",
                    choices=["mean", "attn", "slot", "spatial"],
                    help="Encoder spatial pooling. 'attn' = multi-query "
                         "attention pool fused to one vector (v5.2); 'slot' = "
                         "keep the M-slot axis for z_static (v5.3, object-"
                         "centric); 'spatial' = keep an image-like static grid "
                         "(v5.7, budget-matched, no pool); 'mean' = global "
                         "mean (legacy).")
    ap.add_argument("--static_grid", type=int, default=4,
                    help="Side length of the spatial z_static grid for "
                         "--pool_type spatial. d_static must be divisible by "
                         "static_grid^2 (default 4 -> 4x4 cells; d_static 96 "
                         "-> 6 channels/cell).")
    ap.add_argument("--pool_queries", type=int, default=8,
                    help="Number of learned queries for attention pooling.")
    ap.add_argument("--pool_heads", type=int, default=4,
                    help="Attention heads for attention pooling.")
    ap.add_argument("--decoder_type", type=str, default="linear",
                    choices=["linear", "broadcast", "slot", "spatial", "flow"],
                    help="'broadcast' = spatial-broadcast decoder (v5.2); "
                         "'slot' = per-slot alpha-composited decoder (v5.3, "
                         "requires --pool_type slot); 'spatial' = consume the "
                         "spatial z_static grid (v5.7, requires --pool_type "
                         "spatial); 'linear' = single-Linear lift (legacy).")
    ap.add_argument("--dec_depth", type=int, default=3,
                    help="Conv blocks in the spatial-broadcast decoder.")
    ap.add_argument("--lambda_vic_var", type=float, default=0.0,
                    help="VICReg variance-hinge weight on z_static & z_dyn. "
                         "0 disables. Forces per-dim std >= vic_gamma.")
    ap.add_argument("--lambda_vic_cov", type=float, default=0.0,
                    help="VICReg covariance weight on z_static & z_dyn. "
                         "Decorrelates dims to raise participation ratio.")
    ap.add_argument("--lambda_indep", type=float, default=0.0,
                    help="Cross-code independence weight: penalizes cross-correlation "
                         "between z_static and time-pooled z_dyn -> pushes identity "
                         "OUT of z_dyn (the disentanglement fix). 0 disables.")
    ap.add_argument("--vic_gamma", type=float, default=1.0,
                    help="Target per-dim std for the VICReg variance hinge.")

    ap.add_argument("--lambda_recon",     type=float, default=1.0,
                    help="Weight on latent-space L_recon = MSE(dec(z), chunk). "
                         "Set 0 for VAE-unfrozen runs that supervise in pixel space "
                         "instead (--lambda_pixel>0), so the encoder can't game the "
                         "latent target by collapsing.")
    ap.add_argument("--lambda_pred",      type=float, default=1.0)
    ap.add_argument("--lambda_fwd",       type=float, default=0.1)
    ap.add_argument("--lambda_consist",   type=float, default=1.0)
    ap.add_argument("--lambda_event_aux", type=float, default=0.1)
    ap.add_argument("--lambda_gate",      type=float, default=0.1)
    ap.add_argument("--lambda_attrs",     type=float, default=0.0,
                    help="Exp 1: weight on supervised CE (color+material+shape) "
                         "from z_static via AttrsHead. 0.0 disables.")
    ap.add_argument("--attrs_hidden",     type=int,   default=0,
                    help="AttrsHead trunk width; 0 = linear classifier.")
    ap.add_argument("--infonce_temperature", type=float, default=0.1)
    # ---- loss-balance annealing (latent-only -> pixel-dominant curriculum) ----
    ap.add_argument("--anneal_pixel", action="store_true",
                    help="Curriculum on (lambda_recon, lambda_pixel): start "
                         "latent-only, then ramp to the configured pixel-dominant "
                         "targets once train L_recon goes flat. The --lambda_recon "
                         "and --lambda_pixel values are the PHASE-B targets.")
    ap.add_argument("--anneal_recon_hi", type=float, default=1.0,
                    help="Phase-A lambda_recon (latent anchor before the trigger).")
    ap.add_argument("--anneal_pixel_lo", type=float, default=0.0,
                    help="Phase-A lambda_pixel (~0 = latent only before trigger).")
    ap.add_argument("--anneal_plateau_delta", type=float, default=0.005,
                    help="Per-epoch relative L_recon improvement below which an "
                         "epoch counts as 'flat'.")
    ap.add_argument("--anneal_plateau_patience", type=int, default=6,
                    help="Consecutive flat epochs that trigger the ramp to "
                         "pixel-dominant.")
    ap.add_argument("--anneal_ramp_epochs", type=int, default=8,
                    help="Epochs to linearly ramp from phase-A to phase-B weights "
                         "after the trigger (avoids a gradient spike).")
    # v5.1.3: MAETok-style masked semantic prediction (Fork C)
    ap.add_argument("--lambda_mae", type=float, default=0.0,
                    help="Weight on L_mae_sem (masked DINOv2-feature prediction "
                         "through AuxSemanticDecoder). >0 requires --dino_cache_dir.")
    ap.add_argument("--mask_ratio", type=float, default=0.5,
                    help="Fraction of (T,H,W) latent cells replaced by the learnable "
                         "mask vector for the auxiliary encoder pass.")
    ap.add_argument("--mask_pool_frac", type=float, default=0.75,
                    help="Masked cells are sampled from the top this-fraction of "
                         "cells by channel-variance saliency.")
    ap.add_argument("--mask_tube", action="store_true",
                    help="v5.4 gaze fix: mask whole spatial columns (h,w) across "
                         "ALL t instead of per-cell, removing the temporal-copy "
                         "shortcut that makes per-cell video MAE trivial. ratio/"
                         "pool_frac then apply to the H*W columns.")
    ap.add_argument("--aux_hidden_ch", type=int, default=128,
                    help="Hidden channels of AuxSemanticDecoder.")
    ap.add_argument("--dino_cache_dir", type=str, default="",
                    help="Dir from scripts/cache_dino_patch.py (features.f16.bin + "
                         "index.json), rows aligned to --cache_dir windows.")

    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--val_every",  type=int,   default=2)
    ap.add_argument("--max_videos", type=int,   default=0)
    ap.add_argument("--seed",       type=int,   default=42)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every",  type=int, default=1)
    ap.add_argument("--ckpt_every", type=int, default=5,
                    help="Save an epoch-numbered ckpt_ep{N}.pt every N epochs. "
                         "A rolling last.pt is always saved every epoch for resume.")
    ap.add_argument("--resume", type=str, default="",
                    help="Auto-resume: 'auto' loads <out_dir>/last.pt if present "
                         "(restores models, VAE, optimizer, scheduler, best-val "
                         "tracking and epoch). Pass an explicit path to resume from "
                         "a specific ckpt. Empty = start fresh.")
    ap.add_argument("--max_steps",  type=int, default=0)
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="Stop when val_recon hasn't improved for this many val checks. "
                         "0 disables. Requires val_loader. Best ckpt saved as v5_best.pt.")
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                    help="Minimum absolute improvement on val_recon to count as progress.")
    ap.add_argument("--consist_loss", type=str, default="infonce",
                    choices=["infonce", "mse"],
                    help="Ablation 1: 'mse' uses the original v5 MSE consistency "
                         "instead of contrastive InfoNCE on z_static.")
    ap.add_argument("--shared_trunk", action="store_true",
                    help="Ablation 2: single Conv3d trunk feeds both heads "
                         "(disables independent static/dyn trunks).")
    ap.add_argument("--no_proj", action="store_true",
                    help="Exp 3: drop the W_proj/W_unproj bottleneck in "
                         "ForwardDynamics. Requires d_state == d_dyn.")
    ap.add_argument("--wandb_project", type=str, default="dialga",
                    help="W&B project name. Set to '' to disable W&B.")
    ap.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name. Defaults to basename of --out_dir.")
    ap.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    # v5.1.2 VAE-unfreeze experiments (Exp 2/3/4)
    ap.add_argument("--video_dir", type=str,
                    default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video",
                    help="Path to raw CLEVRER mp4 root. Used whenever the VAE/pixel "
                         "path is active (--unfreeze_vae_* or --lambda_pixel>0).")
    ap.add_argument("--unfreeze_vae_enc", action="store_true",
                    help="Re-encode raw pixels through Wan-VAE encoder with grad. "
                         "Latents from cache are ignored for chunk_obs/pred/b.")
    ap.add_argument("--unfreeze_vae_dec", action="store_true",
                    help="Decode predicted latents through Wan-VAE decoder with grad "
                         "and add L_pixel = MSE(pred_pix, GT_pix).")
    ap.add_argument("--lambda_pixel", type=float, default=0.0,
                    help="Weight on pixel-space L_pixel = MSE(wan_dec(recon), raw_pix). "
                         "Any value >0 LOADS the Wan VAE (frozen unless --unfreeze_vae_* "
                         "is also set) and the raw-pixel dataset, so the frozen control "
                         "(e1) can be supervised in pixel space exactly like e2/e3. "
                         "REQUIRED for enc-only runs to keep latents decodable. "
                         "Default 0 = no VAE, latent-space supervision only.")
    ap.add_argument("--vae_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--vae_dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--vae_lr", type=float, default=0.0,
                    help="Separate LR for VAE params. 0 = use same as --lr.")
    # ---- v5.8 camera-aware disentanglement (moving camera) ----
    ap.add_argument("--use_camera_pose", action="store_true",
                    help="Inject a KNOWN camera trajectory: encoder inverts it, "
                         "decoder re-applies it (MG3.5 Warped-PRoPE). Gate config "
                         "only: pool_type=mean, decoder_type=linear, "
                         "lambda_pixel=0, lambda_mae=0.")
    ap.add_argument("--pose_inject", type=str, default="concat",
                    choices=["concat", "prope"],
                    help="concat = pose embedding into z_dyn head; prope = "
                         "relative-pose-biased temporal attention (stronger).")
    ap.add_argument("--d_pose", type=int, default=32, help="camera-pose channel width")
    ap.add_argument("--pose_dim", type=int, default=3,
                    help="raw per-frame pose params (2D pan gate = 3: tx,ty,log_s)")
    ap.add_argument("--dyn_spatial", action="store_true",
                    help="v5.9 recon fix: z_dyn as a per-frame SPATIAL grid "
                         "(c_dyn, dyn_grid, dyn_grid) instead of a global vector. "
                         "Literature (VidTwin/Hi-VAE/Cosmos): real video needs a "
                         "per-frame spatial axis; a global dyn vector caps recon.")
    ap.add_argument("--dyn_grid", type=int, default=8,
                    help="spatial grid side for dyn_spatial z_dyn (d_dyn must be "
                         "divisible by dyn_grid^2).")
    ap.add_argument("--static_agg", type=str, default="sweep",
                    choices=["world", "sweep", "conv"],
                    help="v5.8 spatial static-grid aggregator over the pose-conditioned "
                         "multi-frame window: 'sweep' = plane-sweep depth recovery "
                         "(the fix, 3x more camera-invariant), 'conv' = legacy "
                         "PoseGridAggregator (pose hurt invariance).")
    ap.add_argument("--lambda_cam_inv", type=float, default=1.0,
                    help="weight of the camera-invariance loss on z_static+z_dyn")
    ap.add_argument("--cam_max_shift", type=float, default=0.4,
                    help="max synthetic pan (grid units, full-clip displacement)")
    ap.add_argument("--cam_max_zoom", type=float, default=0.15,
                    help="max synthetic log-zoom over the clip")
    ap.add_argument("--cam_pose_blind", action="store_true",
                    help="CONTROL arm: same capacity + same warp + same invariance "
                         "loss, but feed ZERO pose (camera withheld). Proves the "
                         "pose content, not the extra params, is what de-contaminates.")
    ap.add_argument("--dataset", type=str, default="clevrer",
                    choices=["clevrer", "droid", "ssv2", "libero"],
                    help="clevrer = CLEVRER wan cache; droid = DROID wrist-cam cache "
                         "(real moving camera, per-chunk pose in the batch).")
    ap.add_argument("--cam_source", type=str, default="synth",
                    choices=["synth", "real"],
                    help="synth = synthetic 2D pan warp (CLEVRER gate); real = use "
                         "the KNOWN per-chunk pose from the batch (DROID). real has "
                         "no un-warped reference so L_cam_inv is inert (readout is "
                         "the offline z_dyn-vs-camera-velocity probe).")
    ap.add_argument("--pose_n_trans", type=int, default=2,
                    help="how many leading pose dims are translation (log-compressed "
                         "on relativise). 2D pan = 2 (tx,ty); DROID SE(3) = 3 (xyz).")
    args = ap.parse_args()
    if args.cam_source == "real" and not args.use_camera_pose:
        raise SystemExit("--cam_source real requires --use_camera_pose")
    if args.use_camera_pose:
        # v5.8: camera pose is wired for the mean-pool baseline (linear decoder)
        # AND the v5.7 spatial grid (SpatialGridDecoder). The toy sweep picked the
        # spatial path: multi-frame pose aggregation on a spatial z_static (the
        # depooled latent), decoder re-applies the pose. See project memory.
        ok = (args.pool_type, args.decoder_type) in (("mean", "linear"), ("spatial", "spatial"))
        if not ok:
            raise SystemExit("--use_camera_pose requires (--pool_type mean --decoder_type linear) "
                             "or (--pool_type spatial --decoder_type spatial)")
        if args.lambda_pixel > 0 or args.lambda_mae > 0:
            raise SystemExit("--use_camera_pose gate: set --lambda_pixel 0 --lambda_mae 0")
    # Load the VAE + raw-pixel dataset whenever any VAE half is unfrozen OR a
    # pixel-space loss is requested (lambda_pixel>0). The latter lets the frozen
    # control (e1) share e2/e3's pixel supervision with both halves frozen.
    args.use_pixels = bool(args.unfreeze_vae_enc or args.unfreeze_vae_dec
                           or args.lambda_pixel > 0.0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb init ----
    use_wandb = bool(args.wandb_project) and _WANDB_OK and args.wandb_mode != "disabled"
    if use_wandb:
        run_name = args.wandb_run_name or out_dir.name
        # Quiet wandb's own dir spam (matplotlib etc.)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        os.environ.setdefault("WANDB_SILENT", "true")
        # Resume the SAME wandb run across requeues: the run id is stashed in a
        # small text file in out_dir on first launch.
        init_kw = dict(project=args.wandb_project, name=run_name,
                       mode=args.wandb_mode, config=vars(args), dir=str(out_dir))
        rid_file = out_dir / "wandb_run_id.txt"
        if args.resume and rid_file.exists():
            prev_id = rid_file.read_text().strip()
            if prev_id:
                init_kw.update(id=prev_id, resume="allow")
        wandb.init(**init_kw)
        try:
            rid_file.write_text(wandb.run.id)
        except Exception:
            pass
        print(f"[wandb] project={args.wandb_project} run={run_name} mode={args.wandb_mode}")
    else:
        print("[wandb] disabled" + (" (not installed)" if not _WANDB_OK else ""))

    # ---- data ----
    if args.lambda_mae > 0.0 and not args.dino_cache_dir:
        raise SystemExit("--lambda_mae > 0 requires --dino_cache_dir")

    def _make_ds(split):
        if args.dataset == "libero":
            from src.data.libero_window import LiberoChunkPairs
            sp = split if args.val_frac > 0 else "train"
            return LiberoChunkPairs(args.cache_dir, split=sp,
                                    max_episodes=args.max_videos)
        kw = dict(seed=args.seed, max_videos=args.max_videos)
        if args.val_frac > 0:
            kw.update(split=split, val_frac=args.val_frac)
        if args.dataset == "droid":
            return DroidChunkPairs(args.cache_dir, **kw)
        if args.dataset == "ssv2":
            from src.data.ssv2_window import SSv2ChunkPairs
            if args.lambda_mae > 0.0 and args.dino_cache_dir:
                kw["dino_cache_dir"] = args.dino_cache_dir
            return SSv2ChunkPairs(args.cache_dir, **kw)
        if args.lambda_mae > 0.0:
            kw.update(dino_cache_dir=args.dino_cache_dir)
        if args.use_pixels:
            return ClevrerChunkPairsWithPixels(args.cache_dir, args.video_dir, **kw)
        return ClevrerChunkPairs(args.cache_dir, **kw)

    if args.val_frac > 0:
        ds_train = _make_ds("train")
        ds_val   = _make_ds("val")
        print(f"[data] train={len(ds_train)} pairs, val={len(ds_val)} pairs "
              f"(val_frac={args.val_frac}, use_pixels={args.use_pixels})")
    else:
        ds_train = _make_ds("all")
        ds_val = None
        print(f"[data] no split: {len(ds_train)} pairs (overfit mode, "
              f"use_pixels={args.use_pixels})")

    pin = (device.type == "cuda")
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=chunk_collate,
                        pin_memory=pin, drop_last=True)
    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=chunk_collate,
                                pin_memory=pin)

    # ---- models ----
    _d_pose = args.d_pose if args.use_camera_pose else 0
    enc = LatentEncoder3D(d_static=args.d_static, d_dyn=args.d_dyn,
                          hidden_ch=args.enc_hidden_ch,
                          shared_trunk=args.shared_trunk,
                          pool_type=args.pool_type,
                          n_queries=args.pool_queries,
                          n_heads=args.pool_heads,
                          static_grid=args.static_grid,
                          chunk_size_lat=args.chunk_size_lat,
                          static_agg=args.static_agg,
                          dyn_spatial=args.dyn_spatial, dyn_grid=args.dyn_grid,
                          d_pose=_d_pose).to(device)
    if args.decoder_type == "broadcast":
        dec = SpatialBroadcastDecoder(d_static=args.d_static, d_dyn=args.d_dyn,
                                      hidden_ch=args.dec_hidden_ch,
                                      chunk_size_lat=args.chunk_size_lat,
                                      depth=args.dec_depth).to(device)
    elif args.decoder_type == "slot":
        if args.pool_type != "slot":
            raise ValueError("--decoder_type slot requires --pool_type slot")
        dec = SlotDecoder(d_static=args.d_static, d_dyn=args.d_dyn,
                          hidden_ch=args.dec_hidden_ch,
                          chunk_size_lat=args.chunk_size_lat,
                          depth=args.dec_depth).to(device)
    elif args.decoder_type == "spatial":
        if args.pool_type != "spatial":
            raise ValueError("--decoder_type spatial requires --pool_type spatial")
        dec = SpatialGridDecoder(d_static=args.d_static, static_grid=args.static_grid,
                                 d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                                 chunk_size_lat=args.chunk_size_lat,
                                 depth=args.dec_depth, d_pose=_d_pose,
                                 dyn_spatial=args.dyn_spatial, dyn_grid=args.dyn_grid).to(device)
    elif args.decoder_type == "flow":
        # v6: rectified-flow generative decoder (VideoFlexTok). Consumes the
        # spatial static grid; trained with a velocity MSE (no VAE in loop).
        if args.pool_type != "spatial":
            raise ValueError("--decoder_type flow requires --pool_type spatial")
        dec = FlowMatchingDecoder(d_static=args.d_static, static_grid=args.static_grid,
                                  d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                                  chunk_size_lat=args.chunk_size_lat,
                                  depth=args.dec_depth).to(device)
    else:
        dec = LatentDecoder(d_static=args.d_static, d_dyn=args.d_dyn,
                            hidden_ch=args.dec_hidden_ch,
                            chunk_size_lat=args.chunk_size_lat,
                            d_pose=_d_pose).to(device)
    print(f"[arch] pool_type={args.pool_type} decoder_type={args.decoder_type} "
          f"vic_var={args.lambda_vic_var} vic_cov={args.lambda_vic_cov}")
    fwd = ForwardDynamics(d_dyn=args.d_dyn, d_state=args.d_state,
                          no_proj=args.no_proj).to(device)
    eh = EventHead(d_dyn=args.d_dyn, d_event=args.d_event).to(device)
    ge = GEvent(d_event=args.d_event, d_dyn=args.d_dyn).to(device)
    gp = GatePredictor(d_dyn=args.d_dyn).to(device)
    ah = AttrsHead(d_static=args.d_static, n_color=N_COLOR, n_material=N_MATERIAL,
                   n_shape=N_SHAPE, hidden=args.attrs_hidden).to(device)
    masker, auxd = None, None
    if args.lambda_mae > 0.0:
        d_feat = int(ds_train._dino_shape[-1])
        masker = ObjectRegionMask(latent_ch=48, ratio=args.mask_ratio,
                                  pool_frac=args.mask_pool_frac,
                                  tube=args.mask_tube).to(device)
        auxd = AuxSemanticDecoder(d_feat=d_feat, d_static=args.d_static,
                                  d_dyn=args.d_dyn, hidden_ch=args.aux_hidden_ch,
                                  chunk_size_lat=args.chunk_size_lat).to(device)
        print(f"[mae] masked semantic prediction ON: d_feat={d_feat} "
              f"ratio={args.mask_ratio} pool_frac={args.mask_pool_frac} "
              f"tube={args.mask_tube} "
              f"aux {sum(p.numel() for p in auxd.parameters())/1e6:.2f}M")
    models = (enc, dec, fwd, eh, ge, gp, ah)
    if masker is not None:
        models = models + (masker, auxd)
    if args.use_camera_pose:
        cc = CameraConditioner(pose_dim=args.pose_dim, d_pose=args.d_pose,
                               mode=args.pose_inject, n_trans=args.pose_n_trans,
                               mix_dim=args.enc_hidden_ch).to(device)
        models = models + (cc,)
        print(f"[cam] camera-pose ON: inject={args.pose_inject} d_pose={args.d_pose} "
              f"blind={args.cam_pose_blind} lambda_cam_inv={args.lambda_cam_inv} "
              f"cc {sum(p.numel() for p in cc.parameters())/1e3:.1f}K")
    n_total = sum(sum(p.numel() for p in m.parameters()) for m in models)
    print(f"[model] enc {sum(p.numel() for p in enc.parameters())/1e3:.1f}K | "
          f"dec {sum(p.numel() for p in dec.parameters())/1e6:.2f}M | "
          f"fwd {sum(p.numel() for p in fwd.parameters())} | "
          f"eh {sum(p.numel() for p in eh.parameters())} | "
          f"ge {sum(p.numel() for p in ge.parameters())} | "
          f"gp {sum(p.numel() for p in gp.parameters())} | "
          f"ah {sum(p.numel() for p in ah.parameters())} | "
          f"total {n_total/1e6:.2f}M")

    # ---- (optional) Wan-VAE with selectively-unfrozen halves ----
    vae = None
    if args.use_pixels:
        vae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                     "float32": torch.float32}[args.vae_dtype]
        print(f"[vae] loading {args.vae_model_id} dtype={args.vae_dtype} "
              f"train_enc={args.unfreeze_vae_enc} train_dec={args.unfreeze_vae_dec}")
        vae = load_wan_vae_trainable(args.vae_model_id, vae_dtype, device,
                                     train_enc=args.unfreeze_vae_enc,
                                     train_dec=args.unfreeze_vae_dec)
        n_vae_train = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        n_vae_total = sum(p.numel() for p in vae.parameters())
        print(f"[vae] {n_vae_total/1e6:.1f}M total, {n_vae_train/1e6:.1f}M trainable")

    params = []
    for m in models:
        params += list(m.parameters())
    if vae is not None:
        vae_params = [p for p in vae.parameters() if p.requires_grad]
        if vae_params:
            vae_lr = args.vae_lr if args.vae_lr > 0 else args.lr
            opt = torch.optim.AdamW([
                {"params": params,     "lr": args.lr},
                {"params": vae_params, "lr": vae_lr},
            ], weight_decay=args.weight_decay)
        else:
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.lr_schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"unknown lr_schedule {args.lr_schedule}")

    history = []
    step = 0
    t0 = time.time()
    best_val_recon = float("inf")
    best_epoch = 0
    val_no_improve = 0

    # ---- auto-resume from latest checkpoint ----
    start_epoch = 1
    if args.resume:
        resume_path = None
        if args.resume == "auto":
            cand = out_dir / "last.pt"
            resume_path = cand if cand.exists() else None
        else:
            cand = Path(args.resume)
            resume_path = cand if cand.exists() else None
        if resume_path is not None:
            print(f"[resume] loading {resume_path}")
            ck = torch.load(resume_path, map_location=device)
            enc.load_state_dict(ck["encoder"])
            dec.load_state_dict(ck["decoder"])
            fwd.load_state_dict(ck["fwd"])
            eh.load_state_dict(ck["event_head"])
            ge.load_state_dict(ck["g_event"])
            gp.load_state_dict(ck["gate_predictor"])
            ah.load_state_dict(ck["attrs_head"])
            if masker is not None and "masker" in ck:
                masker.load_state_dict(ck["masker"])
                auxd.load_state_dict(ck["aux_decoder"])
            if vae is not None and "wan_vae" in ck:
                vae.load_state_dict(ck["wan_vae"])
            if "optimizer" in ck:
                opt.load_state_dict(ck["optimizer"])
            if "scheduler" in ck:
                sched.load_state_dict(ck["scheduler"])
            step = int(ck.get("step", 0))
            best_val_recon = float(ck.get("best_val_recon", best_val_recon))
            best_epoch = int(ck.get("best_epoch", best_epoch))
            val_no_improve = int(ck.get("val_no_improve", val_no_improve))
            if isinstance(ck.get("history"), list):
                history = ck["history"]
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[resume] continuing at epoch {start_epoch}/{args.epochs} "
                  f"(step {step}, best_val_recon={best_val_recon:.5f} @ ep {best_epoch})")
        else:
            print(f"[resume] no checkpoint found (--resume {args.resume!r}) "
                  "— starting fresh")

    if start_epoch > args.epochs:
        print(f"[resume] already at epoch {start_epoch - 1} >= {args.epochs}; "
              "nothing to do.")
        (out_dir / "DONE").write_text(f"epochs={args.epochs}\n")
        if use_wandb:
            wandb.finish()
        return

    # Phase-B targets for the loss-balance curriculum are the configured values;
    # captured once so per-epoch mutation of args.lambda_* can ramp toward them.
    recon_target = float(args.lambda_recon)
    pixel_target = float(args.lambda_pixel)

    for ep in range(start_epoch, args.epochs + 1):
        stage = stage_at_epoch(ep, args.stage1_epochs, args.stage2_epochs)
        anneal_info = None
        if args.anneal_pixel:
            cur_recon, cur_pixel, trig, trig_ep = anneal_lambdas(
                ep, history, args, recon_target, pixel_target)
            args.lambda_recon = cur_recon
            args.lambda_pixel = cur_pixel
            anneal_info = {"lam_recon": cur_recon, "lam_pixel": cur_pixel,
                           "triggered": trig, "trigger_ep": trig_ep}
        sums = {}
        n_batches = 0
        [m.train() for m in models]
        for batch in loader:
            losses = compute_losses(batch, models, args, stage, device, vae=vae)
            total = losses["total"]
            opt.zero_grad(set_to_none=True)
            total.backward()
            all_grad_params = params + (
                [p for p in vae.parameters() if p.requires_grad] if vae is not None else []
            )
            torch.nn.utils.clip_grad_norm_(all_grad_params, 1.0)
            opt.step()
            step += 1

            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n_batches += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break
        sched.step()
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}

        val_metrics = None
        if val_loader is not None and (ep % args.val_every == 0 or ep == args.epochs):
            val_metrics = validate(models, val_loader, args, device, vae=vae)

        if ep % args.log_every == 0 or ep == 1:
            keys = ["recon", "pred", "fwd", "consist", "attrs", "event_aux", "gate"]
            if args.use_pixels and getattr(args, "lambda_pixel", 0.0) > 0.0:
                keys += ["pixel", "pixel_pred"]
            if args.lambda_mae > 0.0:
                keys += ["mae_sem"]
            if args.lambda_vic_var > 0.0 or args.lambda_vic_cov > 0.0:
                keys += ["vic_var", "vic_cov"]
            keys += ["total"]
            train_str = " ".join(f"{k}={avg[k]:.5f}" for k in keys if k in avg)
            diag_str = (f" |z_s_std={avg.get('z_static_std', 0):.3f}"
                        f" |z_d_norm={avg.get('z_dyn_obs_norm', 0):.3f}"
                        f" |z_d_roll={avg.get('z_dyn_roll_norm', 0):.3f}"
                        f" |z_ev={avg.get('z_event_norm', 0):.3f}"
                        f" |gate={avg.get('gate_GT_mean', 0):.3f}")
            val_str = ""
            if val_metrics is not None:
                val_str = (f" | val_recon={val_metrics['recon']:.5f}"
                           f" val_pred={val_metrics['pred']:.5f}"
                           f" val_consist={val_metrics['consist']:.4f}"
                           f" val_z_s_std={val_metrics['z_static_std']:.3f}")
            anneal_str = ""
            if anneal_info is not None:
                tg = (f"TRIG@{anneal_info['trigger_ep']}" if anneal_info["triggered"]
                      else "latent-only")
                anneal_str = (f" |anneal {tg} lam_r={anneal_info['lam_recon']:.3f}"
                              f" lam_px={anneal_info['lam_pixel']:.3f}")
            elapsed = time.time() - t0
            print(f"[ep {ep:3d}/{args.epochs} stage{stage}] {train_str}{diag_str}{val_str}"
                  f"{anneal_str} | lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        history.append({"epoch": ep, "stage": stage, "step": step,
                        "lr": opt.param_groups[0]["lr"],
                        "anneal": anneal_info,
                        "train": avg, "val": val_metrics})

        # ---- wandb log (always log per-epoch, even when no val ran this epoch) ----
        if use_wandb:
            log_row = {
                "epoch": ep, "stage": stage, "step": step,
                "lr": opt.param_groups[0]["lr"],
                "wall_s": time.time() - t0,
                **{f"train/{k}": v for k, v in avg.items()},
            }
            if val_metrics is not None:
                log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_row, step=ep)

        def _build_ckpt():
            ck = {
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "fwd": fwd.state_dict(),
                "event_head": eh.state_dict(),
                "g_event": ge.state_dict(),
                "gate_predictor": gp.state_dict(),
                "attrs_head": ah.state_dict(),
                **({"masker": masker.state_dict(),
                    "aux_decoder": auxd.state_dict()} if masker is not None else {}),
                **({"camera": cc.state_dict()} if args.use_camera_pose else {}),
                "args": vars(args),
                "epoch": ep,
                "step": step,
                # --- resume state ---
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best_val_recon": best_val_recon,
                "best_epoch": best_epoch,
                "val_no_improve": val_no_improve,
                "history": history,
            }
            # Only save VAE state when something was actually trainable.
            if vae is not None and any(p.requires_grad for p in vae.parameters()):
                ck["wan_vae"] = vae.state_dict()
            return ck

        # Rolling resume ckpt every epoch (overwritten) so a preemption/timeout
        # loses at most one epoch. Heavy (VAE bundled) but cheap vs 25 min/epoch.
        ck = _build_ckpt()
        # Atomic write: a preemption landing mid-save would otherwise truncate
        # last.pt and break --resume auto. Write to a temp then os.replace (an
        # atomic rename on POSIX), so last.pt is always a complete checkpoint.
        tmp_ckpt = out_dir / "last.pt.tmp"
        torch.save(ck, tmp_ckpt)
        os.replace(tmp_ckpt, out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        # Organized milestone snapshot every ckpt_every epochs.
        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0 or ep == args.epochs):
            torch.save(ck, out_dir / f"ckpt_ep{ep:04d}.pt")
            torch.save(ck, out_dir / "v5.pt")

        # ---- early stopping + best-by-val ckpt ----
        # Only eligible once stage 2 has started — stage 1 ends with low val_recon
        # because InfoNCE/L_pred haven't activated yet; the stage-2 transition
        # bumps val_recon up briefly, and we don't want best_val_recon to lock
        # onto the pre-contrastive ckpt.
        if val_metrics is not None and stage >= 2:
            cur = val_metrics["recon"]
            if cur < best_val_recon - args.early_stop_min_delta:
                best_val_recon = cur
                best_epoch = ep
                val_no_improve = 0
                torch.save(_build_ckpt(), out_dir / "v5_best.pt")
                print(f"  [best] val_recon={cur:.5f} at ep {ep} -> saved v5_best.pt")
                if use_wandb:
                    wandb.log({"val/best_recon": best_val_recon, "val/best_epoch": best_epoch}, step=ep)
            else:
                val_no_improve += 1
                if args.early_stop_patience > 0 and val_no_improve >= args.early_stop_patience:
                    print(f"[stop] early stop: val_recon hasn't improved by "
                          f">={args.early_stop_min_delta} for {val_no_improve} val checks. "
                          f"Best val_recon={best_val_recon:.5f} at ep {best_epoch}.")
                    break

        if args.max_steps > 0 and step >= args.max_steps:
            print(f"[stop] hit max_steps={args.max_steps}")
            break

    # Final ckpt + history
    torch.save(_build_ckpt(), out_dir / "v5.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    # DONE marker: training reached its natural end (final epoch or early stop).
    # The chained sbatch checks this to know it should stop requeueing.
    (out_dir / "DONE").write_text(f"epoch={ep} best_val_recon={best_val_recon:.5f}\n")
    print(f"\n[done] final: {avg}")
    print(f"[best] val_recon={best_val_recon:.5f} at ep {best_epoch} (v5_best.pt)")
    if use_wandb:
        wandb.run.summary["best_val_recon"] = best_val_recon
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()


if __name__ == "__main__":
    main()
