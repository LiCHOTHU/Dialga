"""
Two-stage trainer for the slot-state Lagrangian pipeline.

Stage 1: encoder + ObjectLagrangian
  - state MSE (encoder pred vs GT positions, masked by visibility)
  - DEL residual on GT triples (masked by visibility triples)
  - one-step solver prediction MSE (teacher forced)
  - SIGReg on the flat latent (sigreg encoder only)

Stage 2: pixel decoder, encoder + Lagrangian frozen
  - per-frame pixel MSE between predicted slot-render and GT frame
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.slot_lagrangian import (
    SlotQueryEncoder,
    LatentSIGRegEncoder,
    CollisionImpulse,
    SlotPixelDecoder,
    apply_collision_impulse_to_positions,
)
from src.model.accel_net import AccelNet, verlet_step
from src.model.event_head import EventHead, build_event_features, dilate_label
from src.model.state_representation import SIGReg
from src.dynamics.self_events import event_soft_from_residual
from src.dynamics.pixel_event_teacher import (
    pixel_event_soft_from_frames,
    motion_centroid_event_soft,
)


def _dynamics_type(cfg):
    """Always 'accel' — AccelNet+Verlet is the only supported dynamics."""
    return "accel"

log = logging.getLogger(__name__)


def _wandb_log(run, payload):
    """No-op when wandb is disabled."""
    if run is not None:
        run.log(payload)


def save_loss_curve(history, out_path, title):
    """Plot per-epoch loss curves to a PNG. history is a dict[str, list[float]].
    Each non-zero series gets its own line; epoch axis is shared.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "epoch" not in history or not history["epoch"]:
        return
    epochs = history["epoch"]
    series = [(k, v) for k, v in history.items()
              if k != "epoch" and any(abs(x) > 0 for x in v)]
    if not series:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    for name, values in series:
        ax.plot(epochs, values, label=name, lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("Loss curve saved to %s", out_path)


# ---------------------------------------------------------------- helpers


def set_seed(seed):
    import random as _random
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_encoder(cfg, attr_dim):
    enc_type = str(cfg.model.encoder_type)
    common = dict(
        image_size=int(cfg.dataset.image_size),
        patch_size=int(cfg.model.patch_size),
        embed_dim=int(cfg.model.embed_dim),
        depth=int(cfg.model.encoder_depth),
        num_heads=int(cfg.model.num_heads),
        mlp_ratio=float(cfg.model.mlp_ratio),
        max_objects=int(cfg.dataset.max_objects),
        attr_dim=int(attr_dim),
        num_state_dims=int(cfg.model.num_state_dims),
        d_static=int(cfg.model.get("d_static", 16)),
    )
    if enc_type == "slot":
        return SlotQueryEncoder(**common)
    if enc_type == "sigreg":
        return LatentSIGRegEncoder(latent_dim=int(cfg.model.latent_dim), **common)
    raise ValueError(f"Unknown encoder_type: {enc_type}")


def encode_window(encoder, frames, attrs):
    """frames: (B, W, 3, H, W); attrs: (B, K, A)
    → positions (B, W, K, D), z_static_per_frame (B, W, K, d_static)
    """
    B, W = frames.shape[:2]
    flat = frames.flatten(0, 1)                                  # (B*W, 3, H, W)
    attrs_rep = attrs.unsqueeze(1).expand(B, W, *attrs.shape[1:]).flatten(0, 1)
    pos, z_static = encoder(flat, attrs_rep)                     # (B*W, K, *)
    return (pos.view(B, W, *pos.shape[1:]),
            z_static.view(B, W, *z_static.shape[1:]))


def pool_z_static(z_static_per_frame, visibility):
    """Visibility-weighted mean over time → one z_static per (video, slot).

    z_static_per_frame: (B, W, K, d_static)
    visibility:         (B, W, K)
    Returns z_static_video: (B, K, d_static).
    """
    w = visibility.float().unsqueeze(-1)                          # (B, W, K, 1)
    num = (z_static_per_frame * w).sum(dim=1)                     # (B, K, d)
    den = w.sum(dim=1).clamp_min(1.0)                             # (B, K, 1)
    return num / den


def static_consistency_loss(per_frame, video, visibility):
    """Per-frame z_static must agree with the video-pooled value.

    per_frame:  (B, W, K, d_static)
    video:      (B, K, d_static)
    visibility: (B, W, K)
    Returns scalar.
    """
    target = video.unsqueeze(1)                                   # (B, 1, K, d)
    diff_sq = (per_frame - target).pow(2).sum(dim=-1)             # (B, W, K)
    w = visibility.float()
    return (diff_sq * w).sum() / w.sum().clamp_min(1.0)


def encode_window_latent(encoder, frames):
    """sigreg-only: (B, W, 3, H, W) → (B, W, latent_dim)"""
    B, W = frames.shape[:2]
    flat = frames.flatten(0, 1)
    z = encoder.latent(flat)                                     # (B*W, L)
    return z.view(B, W, -1)


def training_dynamic_step_slot(lagrangian, q_prev, q_curr, attrs,
                                alpha_prev, alpha_curr, alpha_next,
                                step_alpha=0.1, solver_steps=1, detach_inputs=True,
                                direction_clip=None):
    """
    Predict q_next by descending the *forced* DEL residual

        D2 L_d(q_{k-1}, q_k) + D1 L_d(q_k, q_{k+1}) + Q_k = 0,  Q_k = -γ · v_mid

    Returns q_next: (B, K, D). Falls back to standard DEL when the lagrangian
    has no dissipation.

    direction_clip: per-slot L2-norm cap on the descent direction. None = off.
        Set during eval/rollout to keep the solver finite when a learned V has
        sharp curvature off-manifold. Slots whose ||direction|| > clip are
        rescaled in-place; finite directions and zero directions are untouched.
    """
    q_prev_s = q_prev.detach() if detach_inputs else q_prev
    q_curr_s = q_curr.detach() if detach_inputs else q_curr
    if not q_curr_s.requires_grad:
        q_curr_s = q_curr_s.requires_grad_(True)
    q_next_guess = (2 * q_curr_s - q_prev_s).detach().requires_grad_(True)

    L_prev = lagrangian.lagrangian_pair(q_prev_s, q_curr_s, attrs, alpha_prev, alpha_curr)
    d2_prev = torch.autograd.grad(L_prev.sum(), q_curr_s, create_graph=True)[0]

    gate_2d = alpha_prev * alpha_curr * alpha_next               # (B, K)
    gate = gate_2d.unsqueeze(-1)                                 # (B, K, 1)

    for _ in range(max(int(solver_steps), 1)):
        L_curr = lagrangian.lagrangian_pair(q_curr_s, q_next_guess, attrs, alpha_curr, alpha_next)
        d1_curr = torch.autograd.grad(L_curr.sum(), q_curr_s, create_graph=True)[0]
        v_mid = 0.5 * (q_next_guess - q_prev_s)
        force = lagrangian.dissipation_force(v_mid, attrs, gate_2d)
        residual = (d2_prev + d1_curr + force) * gate
        residual_energy = 0.5 * residual.flatten(1).pow(2).sum(dim=1)
        direction = torch.autograd.grad(residual_energy.sum(), q_next_guess, create_graph=True)[0]
        if direction_clip is not None:
            direction = torch.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
            norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = (float(direction_clip) / norm).clamp_max(1.0)
            direction = direction * scale
        q_next_guess = q_next_guess - step_alpha * direction
    return q_next_guess


def masked_mse(pred, target, mask):
    """pred/target: (..., D); mask: (...,) bool/float. Returns scalar."""
    mask = mask.float()
    diff_sq = (pred - target).pow(2).sum(dim=-1)                 # (...,)
    denom = mask.sum().clamp_min(1.0)
    return (diff_sq * mask).sum() / denom


# ---------------------------------------------------------------- stage 1


def _impulse_loss(impulse, lagrangian, gt_pos, attrs, collisions_per_sample,
                  visib, W, eps=1e-6):
    """
    Train CollisionImpulse on GT pre/post-collision velocities.

    For each event (t, i, j) in each batch sample:
      v_pre  = q[t-1, .] - q[t-2, .]    (uses purely pre frames)
      v_post = q[t+2, .] - q[t+1, .]    (uses purely post frames)
      pred_v_post_i, pred_v_post_j = impulse(q_at_t[i], q_at_t[j], v_pre, attrs, masses)
      MSE(pred, v_post)  averaged across events.

    Returns (loss_tensor, num_events) — loss is zero if no usable events found.
    """
    device = gt_pos.device
    losses = []
    n_events = 0
    masses_b = lagrangian._mass(attrs)                                # (B, K)
    for b, events in enumerate(collisions_per_sample):
        for (t, i, j) in events:
            if t < 2 or t + 2 >= W:
                continue
            if visib[b, t - 2, i] < 0.5 or visib[b, t - 2, j] < 0.5:
                continue
            if visib[b, t + 2, i] < 0.5 or visib[b, t + 2, j] < 0.5:
                continue
            v_pre_i = (gt_pos[b, t - 1, i] - gt_pos[b, t - 2, i]).unsqueeze(0)
            v_pre_j = (gt_pos[b, t - 1, j] - gt_pos[b, t - 2, j]).unsqueeze(0)
            v_post_i = (gt_pos[b, t + 2, i] - gt_pos[b, t + 1, i]).unsqueeze(0)
            v_post_j = (gt_pos[b, t + 2, j] - gt_pos[b, t + 1, j]).unsqueeze(0)
            q_i = gt_pos[b, t, i].unsqueeze(0)
            q_j = gt_pos[b, t, j].unsqueeze(0)
            a_i = attrs[b, i].unsqueeze(0)
            a_j = attrs[b, j].unsqueeze(0)
            m_i = masses_b[b, i:i + 1]
            m_j = masses_b[b, j:j + 1]
            pred_i, pred_j = impulse(q_i, q_j, v_pre_i, v_pre_j, a_i, a_j, m_i, m_j)
            losses.append(((pred_i - v_post_i).pow(2).sum()
                           + (pred_j - v_post_j).pow(2).sum()) * 0.5)
            n_events += 1
    if not losses:
        return gt_pos.new_zeros(()), 0
    return torch.stack(losses).mean(), n_events


def stage1(cfg, device, encoder, lagrangian, impulse, sigreg, event_head, loader, output_dir, wandb_run=None, pixel_decoder=None):
    encoder.train(); lagrangian.train(); impulse.train()
    if sigreg is not None:
        sigreg.train()
    if event_head is not None:
        event_head.train()
    if pixel_decoder is not None:
        pixel_decoder.train()

    params = (list(encoder.parameters())
              + list(lagrangian.parameters())
              + list(impulse.parameters()))
    if event_head is not None:
        params = params + list(event_head.parameters())
    if pixel_decoder is not None:
        params = params + list(pixel_decoder.parameters())
    optim = AdamW(params, lr=float(cfg.training.stage1_lr),
                  weight_decay=float(cfg.training.stage1_weight_decay))
    epochs = int(cfg.training.stage1_epochs)
    sched = CosineAnnealingLR(optim, T_max=max(epochs, 1))

    lambda_state = float(cfg.training.lambda_state)
    # Curriculum on lambda_state: anneal linearly from `lambda_state` to
    # `lambda_state_anneal_to` over `lambda_state_anneal_epochs` epochs, then
    # hold. Used to escape the constant-velocity collapse in the general
    # regime: strong GT-q pull early forces the encoder to reproduce
    # impulsive collision moments; once 2nd-diff has real spikes, the
    # self-event teacher has signal, and lambda_state can decay.
    lambda_state_anneal_to = float(cfg.training.get("lambda_state_anneal_to", lambda_state))
    lambda_state_anneal_epochs = int(cfg.training.get("lambda_state_anneal_epochs", 0))
    lambda_del = float(cfg.training.get("lambda_del", 0.0))
    lambda_solver = float(cfg.training.lambda_solver)
    lambda_sigreg = float(cfg.training.get("lambda_sigreg", 0.0))
    lambda_collision = float(cfg.training.get("lambda_collision", 1.0))
    lambda_static = float(cfg.training.get("lambda_static", 0.0))
    lambda_event = float(cfg.training.get("lambda_event", 0.0))
    event_pos_weight = float(cfg.training.get("event_pos_weight", 50.0))
    event_label_dilation = int(cfg.training.get("event_label_dilation", 3))
    event_input_mode = str(cfg.model.get("event_input_mode", "q"))
    # Domain-portable supervision switches:
    #   event_supervision: "gt"   = BCE vs dilated GT collision_mask (CLEVRER)
    #                      "self" = BCE vs self-generated z-score residual labels
    #   lambda_recon       > 0    = include pixel-reconstruction grounding via
    #                               pixel_decoder (replaces or augments lambda_state)
    event_supervision = str(cfg.training.get("event_supervision", "gt")).lower()
    # Stage-1 pixel-recon weight (separate from Stage-2's lambda_recon).
    lambda_recon = float(cfg.training.get("lambda_recon_stage1", 0.0))
    self_event_z_thresh = float(cfg.training.get("self_event_z_thresh", 1.5))
    self_event_sharpness = float(cfg.training.get("self_event_sharpness", 2.5))
    # Motion-centroid teacher: absolute |Δ²c| threshold. When > 0 it replaces
    # per-window z-scoring (which is statistical noise at T=6 windows). Smooth
    # motion gives |Δ²c| ≈ 0; collisions give ≈ 0.05-0.4 in normalized coords.
    motion_abs_thresh = float(cfg.training.get("motion_abs_thresh", 0.0))
    # Hard-binarize teacher labels at 0.5 (removes soft mid-range mass).
    event_teacher_hard = bool(cfg.training.get("event_teacher_hard", False))
    # Stabilize the motion teacher by anchoring its Gaussian mask at GT
    # positions (instead of encoder q_pred). Decouples teacher quality from
    # encoder drift — labels stay fixed across epochs. The encoder still
    # predicts q for inference; only the *teacher signal* uses GT q.
    motion_teacher_gt_q = bool(cfg.training.get("motion_teacher_gt_q", False))
    motion_pair_augment_distance = float(cfg.training.get("motion_pair_augment_distance", 0.0))
    # InfoNCE-style slot-discrimination loss on z_static. Prevents all slots
    # from collapsing to the same identity (which the static-consistency loss
    # alone cannot prevent).
    lambda_contrastive = float(cfg.training.get("lambda_contrastive", 0.0))
    lambda_pushforward = float(cfg.training.get("lambda_pushforward", 0.0))
    lambda_lipschitz = float(cfg.training.get("lambda_lipschitz", 0.0))
    lambda_rollout_k = float(cfg.training.get("lambda_rollout_k", 0.0))
    rollout_k = int(cfg.training.get("rollout_k", 1))
    solver_alpha = float(cfg.training.get("solver_alpha", 0.1))
    solver_steps = int(cfg.training.get("solver_steps", 1))
    noise_sigma = float(cfg.training.get("noise_sigma", 0.0))
    pushforward_steps = int(cfg.training.get("pushforward_steps", 1))
    grad_clip = float(cfg.training.grad_clip_norm)
    log_interval = int(cfg.training.log_interval)
    pos_norm = float(cfg.dataset.pos_normalize)

    enc_type = str(cfg.model.encoder_type)
    is_sigreg = (enc_type == "sigreg")
    dyn_type = _dynamics_type(cfg)
    is_accel = (dyn_type == "accel")
    if is_accel:
        # DEL/Lipschitz/pushforward/rollout-k are Lagrangian-only.
        lambda_del = 0.0
        lambda_lipschitz = 0.0
        lambda_pushforward = 0.0
        lambda_rollout_k = 0.0

    log.info("Stage 1 — encoder=%s, dynamics=%s, %d epochs, lr=%.1e",
             enc_type, dyn_type, epochs, optim.param_groups[0]["lr"])

    history = {"epoch": [], "total": [], "state": [], "del": [], "solver": [],
               "pushforward": [], "sigreg": [], "collision": [], "static": [],
               "event": [], "recon": [], "contrastive": []}

    for epoch in range(1, epochs + 1):
        # Curriculum: linearly anneal lambda_state from its starting value
        # toward lambda_state_anneal_to over lambda_state_anneal_epochs.
        if lambda_state_anneal_epochs > 0:
            frac = min(1.0, max(0.0, (epoch - 1) / float(lambda_state_anneal_epochs)))
            lambda_state_curr = lambda_state + (lambda_state_anneal_to - lambda_state) * frac
        else:
            lambda_state_curr = lambda_state

        sums = {"state": 0.0, "del": 0.0, "solver": 0.0, "sigreg": 0.0,
                "collision": 0.0, "pushforward": 0.0, "static": 0.0, "total": 0.0,
                "event": 0.0, "recon": 0.0, "contrastive": 0.0}
        steps = 0
        n_coll_events = 0

        for batch in loader:
            frames = batch["frames"].to(device)                   # (B, W, 3, H, W)
            gt_pos = batch["positions"].to(device) / pos_norm     # (B, W, K, D)
            visib = batch["visibility"].to(device).float()        # (B, W, K)
            slot_mask = batch["slot_mask"].to(device).float()     # (B, K) real-object mask
            attrs = batch["attrs"].to(device)                     # (B, K, A)
            coll_mask = batch["collision_mask"].to(device).bool()  # (B, W, K)
            collisions_per_sample = batch["collisions"]            # list[B] of list[(t,i,j)]
            B, W = frames.shape[:2]

            # Dynamics mask: train Lagrangian on every frame for every real
            # object, regardless of camera visibility. CLEVRER provides GT
            # positions for off-camera objects, so the Lagrangian sees the
            # full trajectory (including positions outside camera view).
            dyn_mask = slot_mask.view(B, 1, -1).expand(B, W, -1).contiguous()

            optim.zero_grad(set_to_none=True)

            # 1) encoder predicts (state, z_static_per_frame) on every frame.
            #    Pool z_static over time (visibility-weighted) to get one
            #    per-(video, slot) identity vector that we feed to dynamics
            #    + decoder *in place of GT attrs*.
            q_pred, z_static_per_frame = encode_window(encoder, frames, attrs)
            z_static_video = pool_z_static(z_static_per_frame, visib)  # (B, K, d_static)
            state_loss = masked_mse(q_pred, gt_pos, visib)
            static_loss = static_consistency_loss(z_static_per_frame, z_static_video, visib)

            # 2) DEL residual on GT triples (slot-mask-gated; excludes collisions)
            if is_accel:
                del_loss = q_pred.new_zeros(())
            else:
                del_loss = compute_lagrangian_window_loss(
                    lagrangian, gt_pos, z_static_video, dyn_mask,
                    collision_mask=coll_mask, create_graph=True,
                )

            # 3) one-step solver prediction MSE on every real-object triple.
            #    GNS/MGN recipe: add Gaussian noise (σ = noise_sigma in
            #    normalized position units) to (q_prev, q_curr) inputs but
            #    keep the target clean. This forces the Lagrangian to be
            #    locally stable around the GT manifold and dramatically
            #    extends the stable rollout horizon (Sanchez-Gonzalez 2020,
            #    Pfaff 2021).
            solver_terms = []
            cm_f = coll_mask.float()
            for t in range(1, W - 1):
                dyn_gate = dyn_mask[:, t - 1] * dyn_mask[:, t] * dyn_mask[:, t + 1]
                col_in_triple = (cm_f[:, t - 1] + cm_f[:, t] + cm_f[:, t + 1]) > 0
                gate = dyn_gate * (~col_in_triple).float()
                if gate.sum().item() == 0:
                    continue

                q_prev_in = gt_pos[:, t - 1]
                q_curr_in = gt_pos[:, t]
                if noise_sigma > 0:
                    q_prev_in = q_prev_in + torch.randn_like(q_prev_in) * noise_sigma
                    q_curr_in = q_curr_in + torch.randn_like(q_curr_in) * noise_sigma

                if is_accel:
                    q_next_pred = verlet_step(
                        lagrangian, q_prev_in.detach(), q_curr_in.detach(),
                        z_static_video,
                        dyn_mask[:, t - 1], dyn_mask[:, t],
                    )
                else:
                    q_next_pred = training_dynamic_step_slot(
                        lagrangian,
                        q_prev_in, q_curr_in, z_static_video,
                        dyn_mask[:, t - 1], dyn_mask[:, t], dyn_mask[:, t + 1],
                        step_alpha=solver_alpha, solver_steps=solver_steps,
                        detach_inputs=True,
                    )
                tgt = gt_pos[:, t + 1]                 # CLEAN target
                diff_sq = (q_next_pred - tgt).pow(2).sum(dim=-1) * gate
                denom = gate.sum().clamp_min(1.0)
                solver_terms.append(diff_sq.sum() / denom)
            solver_loss = (torch.stack(solver_terms).mean()
                           if solver_terms else q_pred.new_zeros(()))

            # 3b) Pushforward / k-step rollout loss (Brandstetter 2022).
            #     Roll the solver `pushforward_steps` times from a single GT
            #     triple. Stop gradient on all intermediate steps, backprop
            #     only the final step's MSE against GT. This trains the
            #     Lagrangian on its own (slightly drifted) outputs.
            pushforward_loss = q_pred.new_zeros(())
            if lambda_pushforward > 0 and pushforward_steps >= 2 and W >= 3:
                # need at least pushforward_steps+1 frames after the start
                pf_terms = []
                for t0 in range(0, W - pushforward_steps - 1):
                    # gate by every involved frame being a real-object slot
                    # AND no collision in any of the involved frames
                    pf_gate = dyn_mask[:, t0]
                    coll_in_window = cm_f[:, t0]
                    for s in range(pushforward_steps + 1):
                        pf_gate = pf_gate * dyn_mask[:, t0 + 1 + s]
                        coll_in_window = coll_in_window + cm_f[:, t0 + 1 + s]
                    pf_gate = pf_gate * (coll_in_window < 0.5).float()
                    if pf_gate.sum().item() == 0:
                        continue

                    q_prev = gt_pos[:, t0]
                    q_curr = gt_pos[:, t0 + 1]
                    if noise_sigma > 0:
                        q_prev = q_prev + torch.randn_like(q_prev) * noise_sigma
                        q_curr = q_curr + torch.randn_like(q_curr) * noise_sigma

                    for s in range(pushforward_steps):
                        q_next = training_dynamic_step_slot(
                            lagrangian, q_prev, q_curr, z_static_video,
                            dyn_mask[:, t0 + s], dyn_mask[:, t0 + s + 1],
                            dyn_mask[:, t0 + s + 2],
                            step_alpha=solver_alpha, solver_steps=solver_steps,
                            detach_inputs=True,
                        )
                        if s < pushforward_steps - 1:
                            q_next = q_next.detach()  # stop-grad on intermediate
                        q_prev, q_curr = q_curr, q_next

                    final_pred = q_curr   # = q at frame t0 + pushforward_steps + 1
                    target = gt_pos[:, t0 + pushforward_steps + 1]
                    diff_sq = (final_pred - target).pow(2).sum(dim=-1) * pf_gate
                    denom = pf_gate.sum().clamp_min(1.0)
                    pf_terms.append(diff_sq.sum() / denom)
                if pf_terms:
                    pushforward_loss = torch.stack(pf_terms).mean()

            # 3c-pre) k-step rollout loss with FULL gradients. Roll the solver
            #     `rollout_k` times from a single GT triple; gradient flows
            #     through every step. Trains the model to track GT through
            #     compounding errors. (No stop-grad → unlike pushforward.)
            rollout_k_loss = q_pred.new_zeros(())
            if lambda_rollout_k > 0 and rollout_k >= 2 and W >= rollout_k + 2:
                rk_terms = []
                for t0 in range(0, W - rollout_k - 1):
                    rk_gate = dyn_mask[:, t0]
                    coll_in_window = cm_f[:, t0]
                    for s in range(rollout_k + 1):
                        rk_gate = rk_gate * dyn_mask[:, t0 + 1 + s]
                        coll_in_window = coll_in_window + cm_f[:, t0 + 1 + s]
                    rk_gate = rk_gate * (coll_in_window < 0.5).float()
                    if rk_gate.sum().item() == 0:
                        continue
                    q_prev_in = gt_pos[:, t0]
                    q_curr_in = gt_pos[:, t0 + 1]
                    if noise_sigma > 0:
                        q_prev_in = q_prev_in + torch.randn_like(q_prev_in) * noise_sigma
                        q_curr_in = q_curr_in + torch.randn_like(q_curr_in) * noise_sigma
                    step_losses = []
                    for s in range(rollout_k):
                        q_next = training_dynamic_step_slot(
                            lagrangian, q_prev_in, q_curr_in, z_static_video,
                            dyn_mask[:, t0 + s], dyn_mask[:, t0 + s + 1],
                            dyn_mask[:, t0 + s + 2],
                            step_alpha=solver_alpha, solver_steps=solver_steps,
                            detach_inputs=False,    # let gradients flow through chain
                        )
                        target = gt_pos[:, t0 + s + 2]
                        diff_sq = (q_next - target).pow(2).sum(dim=-1) * rk_gate
                        denom = rk_gate.sum().clamp_min(1.0)
                        step_losses.append(diff_sq.sum() / denom)
                        q_prev_in = q_curr_in
                        q_curr_in = q_next
                    rk_terms.append(torch.stack(step_losses).mean())
                if rk_terms:
                    rollout_k_loss = torch.stack(rk_terms).mean()

            # 3c) Lipschitz-style smoothness penalty on the Lagrangian.
            #     Forces the energy landscape to be globally smoother by
            #     penalizing |∇_q L|² on perturbed states. Reduces extreme
            #     forces when rollout drifts off the GT manifold.
            lipschitz_loss = q_pred.new_zeros(())
            if lambda_lipschitz > 0:
                # sample one frame index and noise around GT
                t_lip = torch.randint(0, W - 1, (1,)).item()
                q_a = (gt_pos[:, t_lip] + torch.randn_like(gt_pos[:, t_lip]) * noise_sigma
                       ).detach().requires_grad_(True)
                q_b = (gt_pos[:, t_lip + 1] + torch.randn_like(gt_pos[:, t_lip + 1]) * noise_sigma
                       ).detach().requires_grad_(True)
                L_pert = lagrangian.lagrangian_pair(
                    q_a, q_b, z_static_video,
                    dyn_mask[:, t_lip], dyn_mask[:, t_lip + 1],
                )
                grad_a = torch.autograd.grad(L_pert.sum(), q_a, create_graph=True)[0]
                grad_b = torch.autograd.grad(L_pert.sum(), q_b, create_graph=True)[0]
                lipschitz_loss = (grad_a.pow(2).sum(dim=-1).mean()
                                  + grad_b.pow(2).sum(dim=-1).mean()) * 0.5

            # 4) optional SIGReg on flat latent
            sigreg_loss = q_pred.new_zeros(())
            if is_sigreg and sigreg is not None and lambda_sigreg > 0:
                z = encode_window_latent(encoder, frames)         # (B, W, L)
                sigreg_loss = sigreg(z.transpose(0, 1))           # SIGReg expects (T, B, D)

            # 5) collision impulse loss on GT events
            collision_loss, ev_n = _impulse_loss(
                impulse, lagrangian, gt_pos, z_static_video,
                collisions_per_sample, visib, W,
            )
            n_coll_events += ev_n

            # 6) event-head loss: BCE between predicted per-(t, slot) event
            #    logits and a target. Two supervision modes:
            #    "gt":   dilated CLEVRER GT collision_mask (CLEVRER-only).
            #    "self": self-generated z-score residual labels (domain-
            #            portable; no annotation needed). The teacher signal
            #            is detached and re-derived each step from q_pred so
            #            it tracks the evolving encoder.
            event_loss = q_pred.new_zeros(())
            if event_head is not None and lambda_event > 0:
                # Decouple head training from encoder convergence: feed GT q
                # to the head (matches the GT-q-anchored teacher labels). At
                # inference, encoder q_pred is fed in. The encoder still gets
                # gradient from state_loss → q_pred → GT q, so at convergence
                # the train/infer distributions match.
                head_q = gt_pos if bool(cfg.training.get(
                    "event_head_train_on_gt_q", False)) else q_pred
                if event_input_mode == "qva":
                    head_input = build_event_features(head_q)
                elif event_input_mode == "qva_nn":
                    head_input = build_event_features(head_q, include_neighbor=True)
                else:
                    head_input = head_q
                event_logits = event_head(head_input, z_static_video)         # (B, W, K)
                if event_supervision == "self":
                    event_target = event_soft_from_residual(
                        q_pred, visib,
                        z_thresh=self_event_z_thresh,
                        sharpness=self_event_sharpness,
                        label_dilation=event_label_dilation,
                    )
                elif event_supervision == "pixel":
                    # First-attempt pixel teacher (Δ²I) — math-wrong; smooth
                    # motion produces large signal (∝ v² · obj''). Kept for
                    # reproducing the failed baseline.
                    event_target = pixel_event_soft_from_frames(
                        frames, q_pred, visib,
                        attention_sigma=float(cfg.training.get("pixel_event_attention_sigma", 0.15)),
                        z_thresh=self_event_z_thresh,
                        sharpness=self_event_sharpness,
                        label_dilation=event_label_dilation,
                    )
                elif event_supervision == "motion":
                    # Math-corrected pixel teacher: 2nd-difference of the
                    # per-slot motion centroid. Constant-velocity motion gives
                    # a centroid linear in t → Δ²c ≈ 0 (proper inertial
                    # baseline). Impulsive collisions give a centroid kink
                    # → Δ²c spikes. Signal is anchored in pixel motion;
                    # encoder cannot smooth it away.
                    teacher_q = gt_pos if motion_teacher_gt_q else q_pred
                    event_target = motion_centroid_event_soft(
                        frames, teacher_q, visib,
                        attention_sigma=float(cfg.training.get("pixel_event_attention_sigma", 0.15)),
                        z_thresh=self_event_z_thresh,
                        sharpness=self_event_sharpness,
                        label_dilation=event_label_dilation,
                        abs_thresh=motion_abs_thresh,
                        hard_binarize=event_teacher_hard,
                        pair_augment_distance=motion_pair_augment_distance,
                    )
                else:
                    event_target = dilate_label(coll_mask, event_label_dilation)
                event_mask = slot_mask.view(B, 1, -1).expand(B, W, -1)        # real slots only
                pos_weight = torch.tensor(event_pos_weight, device=device)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    event_logits, event_target, pos_weight=pos_weight, reduction="none",
                )
                denom = event_mask.sum().clamp_min(1.0)
                event_loss = (bce * event_mask).sum() / denom

            # 7) pixel-reconstruction grounding (replaces GT-q regression in
            #    the domain-portable regime). The slot decoder maps
            #    (q_pred, z_static, visibility) -> RGB, and we MSE against
            #    GT frames. This gives the encoder a grounding signal that
            #    does NOT require GT object positions — only raw video.
            recon_loss = q_pred.new_zeros(())
            if pixel_decoder is not None and lambda_recon > 0:
                # Per-frame decode: flatten (B, W) into a single batch dim
                flat_q = q_pred.reshape(B * W, *q_pred.shape[2:])
                flat_z = z_static_video.unsqueeze(1).expand(-1, W, -1, -1).reshape(B * W, *z_static_video.shape[1:])
                flat_v = visib.reshape(B * W, *visib.shape[2:])
                recon = pixel_decoder(flat_q, flat_z, flat_v)                 # (B*W, 3, H, W)
                recon = recon.view(B, W, *recon.shape[1:])
                recon_loss = torch.nn.functional.mse_loss(recon, frames)

            # 8) slot-contrastive loss on z_static (InfoNCE-style). Forces
            #    different slots to have distinguishable identities so the
            #    encoder cannot collapse all z_i to the same vector. For each
            #    video, positives = (slot i in frame t, slot i pooled vector);
            #    negatives = (slot i, slot j ≠ i pooled vectors).
            contrastive_loss = q_pred.new_zeros(())
            if lambda_contrastive > 0:
                # z_static_per_frame : (B, W, K, d_s)
                # z_static_video     : (B, K, d_s)  — the pooled identity
                # Per-(video, frame), each slot's per-frame z must be most
                # similar to its own pooled vector and dissimilar from the
                # other slots' pooled vectors in the same video.
                z_per = z_static_per_frame                                    # (B, W, K, d_s)
                z_vid = z_static_video                                        # (B, K, d_s)
                z_per_n = torch.nn.functional.normalize(z_per, dim=-1)
                z_vid_n = torch.nn.functional.normalize(z_vid, dim=-1)
                # logits[b, w, i, j] = sim(z_per[b, w, i], z_vid[b, j])
                logits = torch.einsum("bwid,bjd->bwij", z_per_n, z_vid_n) / 0.1
                # target: each (b, w, i) wants j = i
                targets = (torch.arange(logits.shape[-2], device=device)
                           .view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1))
                # Mask: only train on slots present this frame AND real objects.
                slot_real = slot_mask.view(B, 1, -1).expand(B, W, -1)         # (B, W, K)
                ce_mask = (visib * slot_real).bool()                          # (B, W, K)
                if ce_mask.any():
                    logits_flat = logits.view(-1, logits.shape[-1])           # (B*W*K, K)
                    targets_flat = targets.reshape(-1)                        # (B*W*K,)
                    ce_per = torch.nn.functional.cross_entropy(
                        logits_flat, targets_flat, reduction="none",
                    ).view(B, W, -1)
                    contrastive_loss = (ce_per * ce_mask.float()).sum() / ce_mask.float().sum().clamp_min(1.0)

            # Temporal smoothness regularizer on encoder q_pred. Penalizes
            # ||Δ²q_pred||² per (B, t, K), masked by visibility triples.
            # Why: the EventHead's a-channel is Δ²q. With a noisy encoder,
            # frame-to-frame jitter in q_pred swamps the collision signal.
            # Penalizing Δ²q_pred globally encourages smooth trajectories —
            # the state_loss still pulls toward GT positions, so the only
            # acceleration that survives is real collision physics.
            lambda_q_smooth = float(cfg.training.get("lambda_q_smooth", 0.0))
            q_smooth_loss = q_pred.new_zeros(())
            if lambda_q_smooth > 0 and q_pred.shape[1] >= 3:
                q_a = q_pred[:, 2:] - 2 * q_pred[:, 1:-1] + q_pred[:, :-2]    # (B, W-2, K, D)
                vis_triple = visib[:, 2:] * visib[:, 1:-1] * visib[:, :-2]    # (B, W-2, K)
                num = (q_a.pow(2).sum(-1) * vis_triple).sum()
                den = vis_triple.sum().clamp_min(1.0)
                q_smooth_loss = num / den

            total = (lambda_state_curr * state_loss
                     + lambda_del * del_loss
                     + lambda_solver * solver_loss
                     + lambda_sigreg * sigreg_loss
                     + lambda_collision * collision_loss
                     + lambda_static * static_loss
                     + lambda_pushforward * pushforward_loss
                     + lambda_lipschitz * lipschitz_loss
                     + lambda_rollout_k * rollout_k_loss
                     + lambda_event * event_loss
                     + lambda_recon * recon_loss
                     + lambda_contrastive * contrastive_loss
                     + lambda_q_smooth * q_smooth_loss)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optim.step()

            sums["state"] += state_loss.item()
            sums["del"] += del_loss.item()
            sums["solver"] += solver_loss.item()
            sums["sigreg"] += sigreg_loss.item() if isinstance(sigreg_loss, torch.Tensor) else 0.0
            sums["collision"] += collision_loss.item() if isinstance(collision_loss, torch.Tensor) else 0.0
            sums["pushforward"] += pushforward_loss.item() if isinstance(pushforward_loss, torch.Tensor) else 0.0
            sums["static"] += static_loss.item()
            sums["total"] += total.item()
            sums["event"] += event_loss.item() if isinstance(event_loss, torch.Tensor) else 0.0
            sums["recon"] += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0
            sums["contrastive"] += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
            steps += 1

        sched.step()
        avg = {k: v / max(steps, 1) for k, v in sums.items()}
        history["epoch"].append(epoch)
        for k in ("total", "state", "del", "solver", "pushforward", "sigreg", "collision", "static", "event", "recon", "contrastive"):
            history[k].append(avg[k])
        if epoch % log_interval == 0 or epoch == 1:
            log.info(
                "[stage1] ep %3d/%d | total %.5f | state %.6f (λ=%.3f) | solver %.6f | "
                "static %.5f | event %.5f (%s) | recon %.5f | contrast %.5f | coll %.5f (n=%d) | lr %.2e",
                epoch, epochs,
                avg["total"], avg["state"], lambda_state_curr, avg["solver"],
                avg["static"], avg["event"], event_supervision,
                avg["recon"], avg["contrastive"], avg["collision"], n_coll_events,
                optim.param_groups[0]["lr"],
            )
        _wandb_log(wandb_run, {
            "stage1/epoch": epoch,
            "stage1/total": avg["total"],
            "stage1/state": avg["state"],
            "stage1/del": avg["del"],
            "stage1/solver": avg["solver"],
            "stage1/pushforward": avg["pushforward"],
            "stage1/sigreg": avg["sigreg"],
            "stage1/collision": avg["collision"],
            "stage1/static": avg["static"],
            "stage1/event": avg["event"],
            "stage1/recon": avg["recon"],
            "stage1/coll_events": n_coll_events,
            "stage1/lr": optim.param_groups[0]["lr"],
        })

        ckpt_every = int(getattr(cfg.training, "ckpt_every", 10))
        if ckpt_every > 0 and (epoch % ckpt_every == 0 or epoch == epochs):
            ckpt_path = output_dir / "stage1.pt"
            blob = {
                "encoder_state_dict": encoder.state_dict(),
                "lagrangian_state_dict": lagrangian.state_dict(),
                "impulse_state_dict": impulse.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "stage": 1,
                "epoch": epoch,
            }
            if event_head is not None:
                blob["event_head_state_dict"] = event_head.state_dict()
                blob["event_input_mode"] = event_input_mode
                # Make the saved cfg self-describing for test_event_head.py
                D_ = int(cfg.model.num_state_dims)
                if event_input_mode == "qva":
                    head_in = 3 * D_
                elif event_input_mode == "qva_nn":
                    head_in = 3 * D_ + 2
                else:
                    head_in = D_
                blob["config"]["model"]["event_input_mode"] = event_input_mode
                blob["config"]["model"]["event_head_num_state_dims"] = head_in
            if pixel_decoder is not None:
                blob["decoder_state_dict"] = pixel_decoder.state_dict()
                blob["pixel_decoder_jointly_trained"] = True
            torch.save(blob, ckpt_path)
            log.info("Stage 1 checkpoint @ ep %d -> %s", epoch, ckpt_path)
    save_loss_curve(history, output_dir / "stage1_loss.png",
                    title="Stage 1 — encoder + Lagrangian losses")


# ---------------------------------------------------------------- stage 2


def stage2(cfg, device, encoder, lagrangian, impulse, decoder, loader, output_dir, wandb_run=None):
    # freeze encoder + Lagrangian + impulse
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in lagrangian.parameters():
        p.requires_grad_(False)
    for p in impulse.parameters():
        p.requires_grad_(False)
    encoder.eval(); lagrangian.eval(); impulse.eval()
    decoder.train()

    optim = AdamW(decoder.parameters(),
                  lr=float(cfg.training.stage2_lr),
                  weight_decay=float(cfg.training.stage2_weight_decay))
    epochs = int(cfg.training.stage2_epochs)
    sched = CosineAnnealingLR(optim, T_max=max(epochs, 1))

    lambda_recon = float(cfg.training.lambda_recon)
    grad_clip = float(cfg.training.grad_clip_norm)
    log_interval = int(cfg.training.log_interval)
    pos_norm = float(cfg.dataset.pos_normalize)

    log.info("Stage 2 — decoder, %d epochs, lr=%.1e", epochs, optim.param_groups[0]["lr"])

    history = {"epoch": [], "recon": []}

    for epoch in range(1, epochs + 1):
        sums = {"recon": 0.0}
        steps = 0
        for batch in loader:
            frames = batch["frames"].to(device)
            visib = batch["visibility"].to(device).float()
            attrs = batch["attrs"].to(device)
            B, W = frames.shape[:2]

            with torch.no_grad():
                q_pred, z_static_per_frame = encode_window(encoder, frames, attrs)
                z_static_video = pool_z_static(z_static_per_frame, visib)

            optim.zero_grad(set_to_none=True)
            # render every frame and average
            flat_q = q_pred.reshape(B * W, *q_pred.shape[2:])
            cond_rep = z_static_video.unsqueeze(1).expand(
                B, W, *z_static_video.shape[1:]
            ).reshape(B * W, *z_static_video.shape[1:])
            visib_rep = visib.reshape(B * W, *visib.shape[2:])
            recon = decoder(flat_q, cond_rep, visib_rep)              # (B*W, 3, H, W)
            recon = recon.view(B, W, 3, recon.shape[-2], recon.shape[-1])

            loss = F.mse_loss(recon, frames) * lambda_recon
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip)
            optim.step()

            sums["recon"] += loss.item()
            steps += 1

        sched.step()
        avg_recon = sums["recon"] / max(steps, 1)
        history["epoch"].append(epoch)
        history["recon"].append(avg_recon)
        if epoch % log_interval == 0 or epoch == 1:
            log.info("[stage2] ep %3d/%d | recon %.6f | lr %.2e",
                     epoch, epochs, avg_recon,
                     optim.param_groups[0]["lr"])
        _wandb_log(wandb_run, {
            "stage2/epoch": epoch,
            "stage2/recon": avg_recon,
            "stage2/lr": optim.param_groups[0]["lr"],
        })

        ckpt_every = int(getattr(cfg.training, "ckpt_every", 10))
        if ckpt_every > 0 and (epoch % ckpt_every == 0 or epoch == epochs):
            ckpt_path = output_dir / "stage2.pt"
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "lagrangian_state_dict": lagrangian.state_dict(),
                "impulse_state_dict": impulse.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "stage": 2,
                "epoch": epoch,
            }, ckpt_path)
            log.info("Stage 2 checkpoint @ ep %d -> %s", epoch, ckpt_path)
    save_loss_curve(history, output_dir / "stage2_loss.png",
                    title="Stage 2 — pixel decoder loss")


# ---------------------------------------------------------------- inference videos


@torch.no_grad()
def _autoregressive_rollout_eval(lagrangian, impulse, q_seq_gt, alpha_seq, cond,
                                 collisions_per_sample, solver_alpha, solver_steps,
                                 direction_clip=0.5, dynamics_type="lagrangian"):
    """No-grad rollout. Dispatches between Lagrangian-solver and Verlet-accel.

    `cond` is the per-slot conditioning latent fed to dynamics + impulse —
    formerly GT attrs, now z_static_video.

    direction_clip / accel_clip caps per-slot output magnitude so a runaway
    Lagrangian curvature or accel prediction can't drive q_next to inf.
    """
    B, W, K, D = q_seq_gt.shape
    pred = q_seq_gt.clone()
    is_accel = (str(dynamics_type).lower() == "accel")
    for t in range(2, W):
        q_prev = pred[:, t - 2].clone()
        q_curr = pred[:, t - 1]
        if impulse is not None:
            pairs_at_prev = [
                [(0, i, j) for (tt, i, j) in collisions_per_sample[b] if tt == t - 1]
                for b in range(B)
            ]
            if any(len(p) > 0 for p in pairs_at_prev):
                with torch.enable_grad():
                    q_prev = apply_collision_impulse_to_positions(
                        impulse, lagrangian, q_prev, q_curr, cond, pairs_at_prev,
                    ).detach()
        if is_accel:
            q_next = verlet_step(
                lagrangian, q_prev, q_curr, cond,
                alpha_seq[:, t - 2], alpha_seq[:, t - 1],
                accel_clip=direction_clip,
            )
        else:
            with torch.enable_grad():
                q_next = training_dynamic_step_slot(
                    lagrangian, q_prev, q_curr, cond,
                    alpha_seq[:, t - 2], alpha_seq[:, t - 1], alpha_seq[:, t],
                    step_alpha=solver_alpha, solver_steps=solver_steps,
                    detach_inputs=True, direction_clip=direction_clip,
                ).detach()
        q_next = torch.nan_to_num(q_next, nan=0.0, posinf=0.0, neginf=0.0)
        # Inject GT for any currently-invisible slot ("external object" handling)
        invisible_now = (alpha_seq[:, t] < 0.5).unsqueeze(-1)
        q_next = torch.where(invisible_now, q_seq_gt[:, t], q_next)
        pred[:, t] = q_next
    return pred


def save_overfit_scene_videos(cfg, device, encoder, lagrangian, impulse,
                              dataset, output_dir, num_videos=5,
                              rollout_length=32):
    """Render rollout scene videos for the overfit set and write them as GIFs
    under output_dir. Uses eval_state_overlay's scene_animation. No wandb dep.
    """
    # Defer import to avoid circular dependency at module load.
    from scripts.eval_state_overlay import (
        scene_animation, make_clevrer_pixel_proj, build_long_window_dataset,
    )

    pos_norm = float(cfg.dataset.pos_normalize)
    solver_alpha = float(cfg.training.get("solver_alpha", 0.1))
    solver_steps = max(int(cfg.training.get("solver_steps", 1)), 2)

    long_ds = build_long_window_dataset(cfg, window_length=rollout_length)
    seen = {}
    for i in range(len(long_ds)):
        s = long_ds[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= num_videos:
            break
    if not seen:
        for i in range(len(long_ds)):
            s = long_ds[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= num_videos:
                break
    if not seen:
        log.warning("save_overfit_scene_videos: dataset is empty")
        return
    samples = list(seen.values())

    batch = paired_collate(samples)
    frames = batch["frames"].to(device)
    gt_pos = batch["positions"].to(device) / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device)
    collisions = batch["collisions"]
    N, W = frames.shape[:2]

    encoder.eval(); lagrangian.eval(); impulse.eval()
    with torch.no_grad():
        q_enc, z_static_per_frame = encode_window(encoder, frames, attrs)
        z_static_video = pool_z_static(z_static_per_frame, visib)
    rollout_q = _autoregressive_rollout_eval(
        lagrangian, impulse, gt_pos, visib, z_static_video, collisions,
        solver_alpha, solver_steps,
        dynamics_type=_dynamics_type(cfg),
    )

    gt_np = (gt_pos * pos_norm).cpu().numpy()
    roll_np = (rollout_q * pos_norm).cpu().numpy()
    enc_np = (q_enc * pos_norm).cpu().numpy()
    visib_np = visib.cpu().numpy()
    attrs_np = attrs.cpu().numpy()
    H_img = frames.shape[-2]; W_img = frames.shape[-1]
    proj = make_clevrer_pixel_proj((W_img, H_img))

    for i, s in enumerate(samples):
        vid = s["video_id"]
        m = visib_np[i] > 0.5
        if not m.any():
            continue
        ext = max(2.0, float(np.abs(gt_np[i][m]).max()) * 1.1)
        anim_path = output_dir / f"scene_video_{vid}.gif"
        try:
            scene_animation(gt_np[i], roll_np[i], enc_np[i],
                            visib_np[i], attrs_np[i], vid, str(anim_path),
                            world_extent=ext, fps=8,
                            frames_seq=frames[i], pixel_proj=proj)
            log.info("Scene video saved: %s", anim_path)
        except Exception as e:
            log.exception("scene_animation failed for video %s: %s", vid, e)


def log_inference_videos(cfg, device, encoder, lagrangian, impulse, decoder,
                         dataset, wandb_run, num_videos=5, output_dir=None):
    """Render rollout videos for `num_videos` distinct videos.

    Always writes local GIFs to `output_dir/decoder_video_<vid>.gif` (one per
    sampled video) showing rows: GT / encoder-recon / rollout-recon. If
    `wandb_run` is provided, also logs the same clips to wandb.
    """
    pos_norm = float(cfg.dataset.pos_normalize)
    solver_alpha = float(cfg.training.get("solver_alpha", 0.1))
    solver_steps = max(int(cfg.training.get("solver_steps", 1)), 2)

    # Pick at most num_videos distinct video_ids, prefer start_frame == 0
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        vid = s["video_id"]
        if vid not in seen and s["start_frame"] == 0:
            seen[vid] = s
        if len(seen) >= num_videos:
            break
    if len(seen) == 0:
        for i in range(len(dataset)):
            s = dataset[i]
            vid = s["video_id"]
            if vid not in seen:
                seen[vid] = s
            if len(seen) >= num_videos:
                break
    if len(seen) == 0:
        log.warning("log_inference_videos: dataset is empty")
        return
    samples = list(seen.values())

    batch = paired_collate(samples)
    frames = batch["frames"].to(device)
    gt_pos = batch["positions"].to(device) / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device)
    collisions = batch["collisions"]
    N, W = frames.shape[:2]

    encoder.eval(); lagrangian.eval(); impulse.eval(); decoder.eval()
    with torch.no_grad():
        q_pred, z_static_per_frame = encode_window(encoder, frames, attrs)
        z_static_video = pool_z_static(z_static_per_frame, visib)
    rollout_q = _autoregressive_rollout_eval(
        lagrangian, impulse, gt_pos, visib, z_static_video, collisions,
        solver_alpha, solver_steps,
        dynamics_type=_dynamics_type(cfg),
    )
    with torch.no_grad():
        cond_rep = z_static_video.unsqueeze(1).expand(
            N, W, *z_static_video.shape[1:]
        ).reshape(N * W, *z_static_video.shape[1:])
        visib_rep = visib.reshape(N * W, *visib.shape[2:])
        recon = decoder(q_pred.reshape(N * W, *q_pred.shape[2:]), cond_rep, visib_rep)
        recon = recon.view(N, W, 3, recon.shape[-2], recon.shape[-1])
        rec_roll = decoder(rollout_q.reshape(N * W, *rollout_q.shape[2:]), cond_rep, visib_rep)
        rec_roll = rec_roll.view(N, W, 3, rec_roll.shape[-2], rec_roll.shape[-1])

    def to_uint8(x):
        return (((x + 1.0) / 2.0).clamp(0, 1) * 255).to(torch.uint8).cpu()

    payload = {}
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, s in enumerate(samples):
        # stack GT | recon | rollout vertically per frame
        gt_v = to_uint8(frames[i])           # (T, 3, H, W)
        rc_v = to_uint8(recon[i])
        ro_v = to_uint8(rec_roll[i])
        stacked = torch.cat([gt_v, rc_v, ro_v], dim=2)  # along H → 3·H tall

        if wandb_run is not None:
            import wandb
            payload[f"video/{s['video_id']:05d}"] = wandb.Video(
                stacked.numpy(), fps=4, format="mp4",
                caption=f"vid {s['video_id']} | rows: GT / recon / rollout",
            )

        if out_dir is not None:
            # convert (T, 3, H_total, W) → list of (H_total, W, 3) uint8 frames
            frames_np = stacked.permute(0, 2, 3, 1).contiguous().numpy()
            gif_path = out_dir / f"decoder_video_{s['video_id']:05d}.gif"
            try:
                import imageio
                imageio.mimsave(str(gif_path), list(frames_np), duration=0.25, loop=0)
            except Exception:
                # PIL fallback
                from PIL import Image
                imgs = [Image.fromarray(f) for f in frames_np]
                imgs[0].save(str(gif_path), save_all=True, append_images=imgs[1:],
                             duration=250, loop=0, optimize=False)
            saved_paths.append(gif_path)

    state_mse = ((q_pred - gt_pos).pow(2).sum(dim=-1) * visib).sum() / visib.sum().clamp_min(1.0)
    if W > 2:
        diff = (rollout_q[:, 2:] - gt_pos[:, 2:]).pow(2).sum(dim=-1) * visib[:, 2:]
        denom = visib[:, 2:].sum().clamp_min(1.0)
        roll_state_mse = (diff.sum() / denom).item()
        if wandb_run is not None:
            payload["eval/rollout_state_mse_world"] = roll_state_mse * pos_norm ** 2
    if wandb_run is not None:
        payload["eval/encoder_state_mse_world"] = state_mse.item() * pos_norm ** 2
        payload["eval/recon_pixel_mse"] = F.mse_loss(recon, frames).item()
        if W > 2:
            payload["eval/rollout_pixel_mse"] = F.mse_loss(rec_roll[:, 2:], frames[:, 2:]).item()
        wandb_run.log(payload)
        log.info("Logged %d inference videos to wandb.", len(samples))
    if saved_paths:
        log.info("Saved %d decoder inference videos to %s", len(saved_paths), out_dir)


# ---------------------------------------------------------------- main


@hydra.main(version_base=None, config_path="../conf", config_name="config_slot")
def main(cfg: DictConfig):
    set_seed(int(cfg.training.seed))
    device = torch.device(str(cfg.training.device) if torch.cuda.is_available() else "cpu")
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    log.info("Output dir: %s", output_dir)
    log.info("Encoder type: %s", cfg.model.encoder_type)

    wandb_run = None
    wandb_cfg = cfg.get("wandb", None)
    if wandb_cfg is not None and bool(wandb_cfg.get("enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project=str(wandb_cfg.get("project", "dialga")),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            group=wandb_cfg.get("group", None),
            job_type=str(wandb_cfg.get("job_type", "train_slot")),
            mode=str(wandb_cfg.get("mode", "online")),
            tags=list(wandb_cfg.get("tags", []) or []),
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        log.info("wandb run: %s", wandb_run.url if wandb_run is not None else "?")

    dataset = ClevrerPairedDataset(
        data_dir=str(cfg.dataset.data_dir),
        annotation_dir=str(cfg.dataset.annotation_dir),
        split=str(cfg.dataset.split),
        window_length=int(cfg.training.window_length),
        frames_per_video=int(cfg.dataset.video_num_frames),
        windows_per_video=int(cfg.training.windows_per_video),
        max_videos=int(cfg.training.max_videos),
        max_objects=int(cfg.dataset.max_objects),
        coordinate_mode=str(cfg.dataset.coordinate_mode),
        image_size=int(cfg.dataset.image_size),
        seed=int(cfg.training.seed),
    )
    log.info("Dataset: %d windows over %d videos", len(dataset),
             int(cfg.training.max_videos) if int(cfg.training.max_videos) > 0
             else "?")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=paired_collate,
        persistent_workers=int(cfg.training.num_workers) > 0,
    )

    attr_dim = dataset.attr_dim
    d_static = int(cfg.model.get("d_static", 16))
    encoder = build_encoder(cfg, attr_dim).to(device)
    dyn_type = "accel"
    # Dynamics + impulse + decoder are conditioned on z_static (dim=d_static),
    # not on GT attrs. We construct each module with attr_dim=d_static so the
    # internal MLPs accept the static-identity latent.
    dynamics = AccelNet(
        num_state_dims=int(cfg.model.num_state_dims),
        attr_dim=d_static,
        hidden=int(cfg.model.lagrangian_hidden),
        use_pair=bool(cfg.model.lagrangian_use_pair),
    ).to(device)
    lagrangian = dynamics  # legacy variable name kept for downstream call sites
    impulse = CollisionImpulse(
        attr_dim=d_static,
        hidden=int(cfg.model.get("impulse_hidden", 64)),
    ).to(device)

    sigreg = None
    if str(cfg.model.encoder_type) == "sigreg":
        sigreg = SIGReg(
            knots=int(cfg.training.get("sigreg_knots", 17)),
            num_proj=int(cfg.training.get("sigreg_num_proj", 256)),
        ).to(device)

    # Optional event head: per-slot temporal-conv event detector trained
    # against CLEVRER GT collision_mask (lambda_event=0 disables it).
    event_head = None
    if float(cfg.training.get("lambda_event", 0.0)) > 0:
        event_mode = str(cfg.model.get("event_input_mode", "q"))
        D_in = int(cfg.model.num_state_dims)
        if event_mode == "qva":
            head_in_state_dims = 3 * D_in
        elif event_mode == "qva_nn":
            head_in_state_dims = 3 * D_in + 2
        else:
            head_in_state_dims = D_in
        event_head = EventHead(
            num_state_dims=head_in_state_dims,
            d_static=d_static,
            hidden=int(cfg.model.get("event_hidden", 64)),
            kernel_size=int(cfg.model.get("event_kernel", 5)),
            depth=int(cfg.model.get("event_depth", 2)),
        ).to(device)
        log.info("EventHead params: %d  (input_mode=%s, in_state_dims=%d)",
                 sum(p.numel() for p in event_head.parameters()),
                 event_mode, head_in_state_dims)

    log.info("Dynamics type   : %s", dyn_type)
    log.info("Encoder params  : %d", sum(p.numel() for p in encoder.parameters()))
    log.info("Dynamics params : %d", sum(p.numel() for p in lagrangian.parameters()))
    log.info("Impulse params  : %d", sum(p.numel() for p in impulse.parameters()))

    # Build pixel decoder up-front when lambda_recon_stage1 > 0 so Stage 1
    # can use it as the pixel-reconstruction grounding signal (replaces GT-q
    # regression in the domain-portable regime).
    lambda_recon_cfg = float(cfg.training.get("lambda_recon_stage1", 0.0))
    decoder = SlotPixelDecoder(
        num_state_dims=int(cfg.model.num_state_dims),
        attr_dim=d_static,
        max_objects=int(cfg.dataset.max_objects),
        image_size=int(cfg.dataset.image_size),
        hidden=int(cfg.model.decoder_hidden),
        slot_embed=int(cfg.model.decoder_slot_embed),
        num_blocks=int(cfg.model.decoder_blocks),
        grid_size=int(cfg.model.decoder_grid_size),
    ).to(device)
    log.info("Decoder params  : %d", sum(p.numel() for p in decoder.parameters()))

    pixel_decoder_for_stage1 = decoder if lambda_recon_cfg > 0 else None
    stage1(cfg, device, encoder, lagrangian, impulse, sigreg, event_head, loader, output_dir,
           wandb_run=wandb_run, pixel_decoder=pixel_decoder_for_stage1)

    stage2(cfg, device, encoder, lagrangian, impulse, decoder, loader, output_dir,
           wandb_run=wandb_run)

    # always-on local-disk overfit scene videos under output_dir
    try:
        save_overfit_scene_videos(cfg, device, encoder, lagrangian, impulse,
                                  dataset, output_dir, num_videos=5,
                                  rollout_length=int(cfg.dataset.video_num_frames))
    except Exception as e:
        log.exception("Scene-video saving failed: %s", e)

    try:
        log_inference_videos(cfg, device, encoder, lagrangian, impulse,
                             decoder, dataset, wandb_run, num_videos=5,
                             output_dir=output_dir)
    except Exception as e:
        log.exception("Inference video logging failed: %s", e)

    if wandb_run is not None:
        wandb_run.finish()

    log.info("Training finished.")


if __name__ == "__main__":
    main()
