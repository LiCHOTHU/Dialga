"""3-component trajectory latent (Slot-rSLDS form).

Decomposes a video clip into three structurally-distinct factors. Following
the formulation aligned with Switching Linear Dynamical Systems (rSLDS,
Linderman 2017) and Marked Temporal Point Processes (Mei-Eisner 2017):

  z_static     : (B, K, d_static)        — slot identity, time-invariant
  z_dyn        : (B, T, K, d_dyn)        — per-slot per-frame smooth state
  event_logits : (B, T, K) + (B, K)      — birth-time logits per slot,
                                            plus a "never born" null logit;
                                            jointly softmaxed over (T+1) per slot.
                                            One birth event per slot at most.

The visibility α[k, t] ∈ [0, 1] is derived from the cumulative birth
probability: α[k, t] = Σ_{s ≤ t} P(birth_k = s). Slot k contributes nothing
to the decoder while α[k, t] = 0. The structural constraint is:
  Σ_t event_prob[k, t] + null_prob[k] = 1   for each (video, slot)

Scene-level constants (background, lighting, camera) are absorbed into the
frozen-VAE-decoder's weights for a single scene class. This matches the
design of DINOSAUR/SAVi/Slot-Contrast.

Mapped onto cognitive memory systems:
  z_static ~ semantic memory (object identity)
  z_dyn    ~ world model / procedural (rSLDS continuous state)
  event    ~ episodic memory (a discrete birth-time mark — point process)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------- encoder


class TrajectoryEncoder(nn.Module):
    """Bidirectional encoder over VAE-latent video, emits three components.

    Input  : VAE latent (B, C, T, H, W).
             Optional `frame_mask: (B, T) bool` — True = visible to encoder.
             Masked frames have their spatial tokens replaced by a learned
             mask token before transformer encoding.
    Output : (z_static, z_dyn, event)
    """

    def __init__(
        self,
        latent_ch: int = 48,
        K: int = 8,
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 4,
        T_max: int = 32,
        spatial_size: int = 8,
        d_static: int = 16,
        d_dyn: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = int(K)
        self.d_model = int(d_model)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.spatial_size = int(spatial_size)

        S = self.spatial_size * self.spatial_size
        self.input_proj = nn.Linear(latent_ch, d_model)
        self.temporal_pos = nn.Embedding(T_max, d_model)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, S, d_model))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Static slot queries (K time-invariant) and per-frame slot queries.
        # Reverting from Iter 9's DC-AC (which over-restricted both channels).
        # Disentanglement here comes from the DECODER-side time-blinding
        # (literature consensus per Video Autoencoder ICCV'21 / DiViD 2025).
        self.slot_queries = nn.Parameter(torch.zeros(1, self.K, d_model))
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        self.frame_slot_queries = nn.Parameter(torch.zeros(1, 1, self.K, d_model))
        nn.init.trunc_normal_(self.frame_slot_queries, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, activation="gelu",
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Output heads
        self.static_head = nn.Linear(d_model, d_static)
        self.dyn_head = nn.Linear(d_model, d_dyn)
        # Event head: scalar logit per (slot, frame) for the birth-time softmax
        self.event_logit_head = nn.Linear(d_model, 1)
        # Null head: scalar "never-born" logit per slot (from the time-pooled
        # static token, since the decision is per-slot)
        self.null_logit_head = nn.Linear(d_model, 1)

    def forward(self, wan_latent: torch.Tensor, frame_mask: torch.Tensor | None = None):
        """
        wan_latent : (B, C, T, H, W)
        frame_mask : (B, T) bool, True = visible to encoder.

        Returns:
            z_static     : (B, K, d_static)
            z_dyn        : (B, T, K, d_dyn)
            event_logits : (B, T, K)        — per-frame birth logit
            null_logits  : (B, K)           — per-slot "never born" logit

        Use `event_to_alpha(event_logits, null_logits)` to get the softmax
        birth probability and cumulative visibility α.
        """
        B, C, T, H, W = wan_latent.shape
        S = H * W
        assert H == W == self.spatial_size, f"expected H=W={self.spatial_size}, got {H}x{W}"

        x = wan_latent.permute(0, 2, 3, 4, 1).reshape(B, T, S, C)
        x = self.input_proj(x)
        x = x + self.spatial_pos
        t_idx = torch.arange(T, device=x.device)
        x = x + self.temporal_pos(t_idx)[None, :, None, :]

        if frame_mask is not None:
            vis = frame_mask.to(x.dtype).view(B, T, 1, 1)
            x = vis * x + (1.0 - vis) * self.mask_token

        static_slot_q = self.slot_queries.expand(B, -1, -1)                  # (B, K, d)
        frame_slot_q = self.frame_slot_queries.expand(B, T, -1, -1)
        frame_slot_q = frame_slot_q + self.temporal_pos(t_idx)[None, :, None, :]

        patch_tokens = x.reshape(B, T * S, -1)
        frame_slot_tokens = frame_slot_q.reshape(B, T * self.K, -1)
        tokens = torch.cat([static_slot_q, frame_slot_tokens, patch_tokens], dim=1)

        tokens = self.encoder(tokens)
        tokens = self.out_norm(tokens)

        offset = 0
        static_out = tokens[:, offset:offset + self.K]; offset += self.K
        frame_slot_out = tokens[:, offset:offset + T * self.K].reshape(B, T, self.K, -1)

        z_static = self.static_head(static_out)                              # (B, K, d_static)
        z_dyn = self.dyn_head(frame_slot_out)                                # (B, T, K, d_dyn)
        event_logits = self.event_logit_head(frame_slot_out).squeeze(-1)     # (B, T, K)
        null_logits = self.null_logit_head(static_out).squeeze(-1)           # (B, K)

        return z_static, z_dyn, event_logits, null_logits


def event_to_alpha(event_logits: torch.Tensor, null_logits: torch.Tensor,
                   use_null: bool = True,
                   hard: bool = False,
                   tau: float = 0.5):
    """Softmax-over-(T+1 or T) per slot, then cumulative-sum → visibility α.

    event_logits : (B, T, K)
    null_logits  : (B, K)
    use_null     : if False, the null option is disabled (softmax over T only).
    hard         : if True, use straight-through Gumbel-softmax → α is discrete
                   {0, 1}, removing the "soft-α with empty z_dyn" escape hatch
                   that allowed the model to avoid event localization.
                   Use during training only; at inference, hard=False gives
                   a smooth attention map you can interpret.
    tau          : Gumbel softmax temperature. Lower = sharper. 0.5 is typical.

    Returns:
        event_prob : (B, T, K)
        null_prob  : (B, K)
        alpha      : (B, T, K) — cumulative visibility; discrete {0,1} when hard.
    """
    B, T, K = event_logits.shape
    if use_null:
        combined = torch.cat([event_logits, null_logits.unsqueeze(1)], dim=1)  # (B, T+1, K)
    else:
        combined = event_logits
    if hard:
        # Gumbel-softmax along the time axis (dim=1). Straight-through gives
        # exact one-hot forward + soft gradient backward.
        probs = F.gumbel_softmax(combined, tau=tau, hard=True, dim=1)
    else:
        probs = F.softmax(combined, dim=1)
    if use_null:
        event_prob = probs[:, :T]
        null_prob = probs[:, T]
    else:
        event_prob = probs
        null_prob = torch.zeros(B, K, device=event_logits.device, dtype=event_logits.dtype)
    alpha = torch.cumsum(event_prob, dim=1)                                    # ∈ {0,1} when hard
    return event_prob, null_prob, alpha


# -------------------------------------------------------------------- predictor


class DynPredictor(nn.Module):
    """Per-slot temporal predictor for z_dyn at masked positions.

    Tiny MLP per slot. The encoder's mask_token already lets the transformer
    estimate z_dyn at masked frames implicitly, but having a dedicated
    predictor gives a clean training signal (a direct predict_loss term)
    and a usable inference module for autoregressive rollout.

    Used optionally: when fill_z_dyn() is called, masked positions are
    overwritten with the predictor's output.
    """

    def __init__(self, d_dyn: int, d_static: int, hidden: int = 128,
                 context: int = 2):
        super().__init__()
        self.d_dyn = int(d_dyn)
        self.context = int(context)
        in_dim = self.context * d_dyn + d_static
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d_dyn),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_dyn: torch.Tensor, z_static: torch.Tensor) -> torch.Tensor:
        """z_dyn: (B, T, K, d_dyn). Returns predicted z_dyn same shape.

        For t < context the residual just returns the input (predictor has
        no past). Loss should be computed on t >= context only.
        """
        B, T, K, D = z_dyn.shape
        ctx_parts = []
        for s in range(1, self.context + 1):
            pad = z_dyn.new_zeros(B, s, K, D)
            past = torch.cat([pad, z_dyn[:, :T - s]], dim=1)
            ctx_parts.append(past)
        ctx = torch.cat(ctx_parts, dim=-1)                              # (B, T, K, C*D)
        zs = z_static.unsqueeze(1).expand(B, T, K, -1)
        x = torch.cat([ctx, zs], dim=-1)
        delta = self.net(x)
        # Predict from most recent past + learned delta.
        most_recent = ctx_parts[0]
        return most_recent + delta


def fill_z_dyn(z_dyn: torch.Tensor, predictor: DynPredictor,
               z_static: torch.Tensor,
               frame_mask: torch.Tensor) -> torch.Tensor:
    """Replace z_dyn at masked frames with predictor output."""
    z_pred = predictor(z_dyn, z_static)
    m = frame_mask.to(z_dyn.dtype).unsqueeze(-1).unsqueeze(-1)          # (B, T, 1, 1)
    return m * z_dyn + (1.0 - m) * z_pred


# -------------------------------------------------------------------- decoder


class TrajectoryDecoder(nn.Module):
    """Three components → VAE latent tokens.

    Reconstructs the full VAE latent (B, C, T, H, W) so the frozen VAE
    decoder can render pixels. Broadcasts z_static over T and concatenates
    with per-frame (z_dyn, event) to form slot context. A bank of (T*H*W)
    spatial-temporal query tokens cross-attends to the slot context.
    """

    def __init__(
        self,
        latent_ch: int = 48,
        K: int = 8,
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 4,
        T_max: int = 32,
        spatial_size: int = 8,
        d_static: int = 16,
        d_dyn: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = int(K)
        self.spatial_size = int(spatial_size)
        self.d_model = int(d_model)
        S = self.spatial_size * self.spatial_size

        # Project each slot's per-frame bundle (static + dyn + α) to d_model.
        # α (visibility, scalar) gates the slot's contribution; "not born yet"
        # slots have α ≈ 0 and pass near-zero tokens to the decoder.
        in_dim = d_static + d_dyn + 1
        self.slot_proj = nn.Linear(in_dim, d_model)
        # NOTE: NO learnable temporal positional embedding on the decoder side.
        # Following Video Autoencoder (Lai+, ICCV 2021) and DiViD (2025):
        # any temporal info the decoder produces MUST flow from z_dyn[t],
        # which is the only per-frame-varying signal it receives. We enforce
        # this via a block-diagonal cross-attention mask (query at time t
        # only attends to slot tokens at time t) — no leakage path.
        self.slot_pos = nn.Embedding(self.K, d_model)

        self.query_spatial_pos = nn.Parameter(torch.zeros(1, 1, S, d_model))
        nn.init.trunc_normal_(self.query_spatial_pos, std=0.02)
        self.query_init = nn.Parameter(torch.zeros(1, 1, S, d_model))
        nn.init.trunc_normal_(self.query_init, std=0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, activation="gelu",
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_static: torch.Tensor, z_dyn: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """
        z_static : (B, K, d_static)        — slot identity
        z_dyn    : (B, T, K, d_dyn)        — per-slot per-frame state
        alpha    : (B, T, K)               — cumulative visibility ∈ [0, 1]
        Returns VAE latent (B, C, T, H, W).

        Slot contributions are gated by α at the token level: a "not born yet"
        slot has α≈0 and its slot_token vanishes; once born (α→1), the slot
        carries full identity+state.
        """
        B, T, K, _ = z_dyn.shape
        S = self.spatial_size * self.spatial_size

        zs = z_static.unsqueeze(1).expand(B, T, K, -1)
        slot_bundle = torch.cat([zs, z_dyn, alpha.unsqueeze(-1)], dim=-1)  # (B, T, K, d_static+d_dyn+1)
        slot_tokens = self.slot_proj(slot_bundle)                          # (B, T, K, d)
        # α-gating at the slot-token level.
        slot_tokens = slot_tokens * alpha.unsqueeze(-1)
        k_idx = torch.arange(K, device=z_dyn.device)
        # slot identity within frame — but NO temporal_pos (would let decoder
        # know what time it's rendering without consulting z_dyn).
        slot_tokens = slot_tokens + self.slot_pos(k_idx)[None, None, :, :]
        slot_ctx = slot_tokens.reshape(B, T * K, -1)

        # Output queries: spatial position only, NO temporal position.
        # Queries at different times are identical content-wise; the
        # block-diagonal cross-attention mask routes each frame's queries
        # to its own slot tokens, so per-frame variation in the output
        # can ONLY come from z_dyn[t] (and α[t]) — the literature consensus
        # fix per Video Autoencoder ICCV'21 / DiViD 2025.
        q = self.query_init.expand(B, T, S, -1)
        q = q + self.query_spatial_pos
        q_flat = q.reshape(B, T * S, -1)

        # Build block-diagonal cross-attention mask:
        # memory_mask[i, j] = True (ignore) iff t_query(i) != t_key(j)
        # Shape (T*S, T*K) bool. PyTorch's `memory_mask` accepts this.
        t_q = torch.arange(T, device=z_dyn.device).repeat_interleave(S)     # (T*S,)
        t_k = torch.arange(T, device=z_dyn.device).repeat_interleave(K)     # (T*K,)
        memory_mask = t_q.unsqueeze(1) != t_k.unsqueeze(0)                   # (T*S, T*K)

        x = self.decoder(tgt=q_flat, memory=slot_ctx, memory_mask=memory_mask)
        x = self.out_norm(x)
        out = self.out_proj(x)

        H = W = self.spatial_size
        out = out.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return out


# -------------------------------------------------------------------- heads


class AttrsHead(nn.Module):
    """Supervised per-slot identity head: predict (color, material, shape) from z_static.

    Used by the iter-22c diagnostic (option C in the iter-22 plan) — direct
    supervision on the GT attrs to test whether the encoder *can* route
    identity into z_static when forced to, regardless of whether DINO-CLS is
    a sufficient teacher.
    """

    def __init__(self, d_static: int, n_color: int = 8, n_material: int = 2,
                 n_shape: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_static, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_color + n_material + n_shape),
        )
        self.splits = (n_color, n_material, n_shape)

    def forward(self, z_static: torch.Tensor):
        """z_static: (B, K, d_static) → (logits_color, logits_material, logits_shape)."""
        x = self.net(z_static)
        return torch.split(x, self.splits, dim=-1)


# -------------------------------------------------------------------- loss


def slot_contrast_loss(z_static_a: torch.Tensor, z_static_b: torch.Tensor,
                       tau: float = 0.1) -> torch.Tensor:
    """CoSA-style slot-contrastive (Manasyan+ CVPR 2025, arXiv 2412.14295).

    Two views of the same video (e.g., different random mask patterns)
    produce z_static_a, z_static_b of shape (B, K, d_static).
    Positive pair: same (b, k) across views. Negative: any other (b', k', view').

    Implementation follows the paper:
      - L2-normalize both views
      - cosine similarity, temperature τ=0.1
      - InfoNCE cross-entropy with anchor=view1, positive=view2 at matching (b,k)
    """
    B, K, d = z_static_a.shape
    a = F.normalize(z_static_a.reshape(B * K, d), dim=-1)
    b = F.normalize(z_static_b.reshape(B * K, d), dim=-1)
    bank = torch.cat([a, b], dim=0)                                     # (2BK, d)
    logits = (a @ bank.T) / tau                                          # (BK, 2BK)
    # mask out self-similarity in view a
    BK = B * K
    self_mask = torch.eye(BK, device=a.device, dtype=torch.bool)
    full_self = torch.zeros(BK, 2 * BK, dtype=torch.bool, device=a.device)
    full_self[:, :BK] = self_mask
    logits = logits.masked_fill(full_self, float("-inf"))
    pos_idx = torch.arange(BK, device=a.device) + BK                    # match in view2
    return F.cross_entropy(logits, pos_idx)


def event_from_predict_error(z_dyn: torch.Tensor,
                             predictor: "DynPredictor",
                             z_static: torch.Tensor,
                             beta: float = 0.9,
                             gamma: float = 1.5) -> torch.Tensor:
    """Soft event-boundary target from EMA-normalized predictor residual
    (Reynolds-Zacks-Braver 2007 / Aakur-Sarkar CVPR 2019 style).

    Returns a (B, T, K) tensor in [0, 1] suitable as a target for an
    event head (BCE supervised), or directly as the gate magnitude itself.

    The signal is differentiable in `z_dyn` but the running stats (mean/std)
    are detached so loss doesn't push z_dyn toward zero variance.
    """
    with torch.no_grad():
        # Predict z_dyn[t+1] from z_dyn[<=t]. Use the predictor's output
        # WITHOUT z_dyn at the current position (it sees only past context).
        z_pred = predictor(z_dyn, z_static)                             # (B, T, K, d_dyn)
    err = (z_dyn.detach() - z_pred).pow(2).mean(dim=-1)                 # (B, T, K)
    mu = err.mean(dim=1, keepdim=True)                                   # (B, 1, K)
    sd = err.std(dim=1, keepdim=True) + 1e-6
    z = (err - mu) / sd                                                  # z-score per-(batch, slot)
    return torch.sigmoid(z - gamma)                                      # (B, T, K) ∈ (0,1)


def trajectory_loss(z_static, z_dyn, event_prob, null_prob, alpha, recon, target,
                    lambda_smooth: float = 0.1,
                    lambda_entropy: float = 0.01,
                    lambda_vicreg: float = 0.01,
                    lambda_diff: float = 0.0,
                    lambda_alpha_sparse: float = 0.0,
                    lambda_slot_bw: float = 0.0):
    """Single-objective training loss for the rSLDS-form model.

    L = MSE(recon, target)
      + λ_diff         · MSE(Δ_t recon, Δ_t target)   (PV-VAE L_Diff)
      + λ_smooth       · α-weighted ‖Δ² z_dyn‖²       (smoothness on live slots)
      + λ_entropy      · H(birth softmax per slot)    (peaked birth time)
      + λ_vicreg       · (var + cov on z_static)
      + λ_alpha_sparse · mean(α[t, k])                (PUSH α=0 when slot
                                                       isn't needed → forces
                                                       unsupervised localization
                                                       of mid-clip entries)
    """
    recon_loss = F.mse_loss(recon, target)

    # PV-VAE L_Diff: MSE on the temporal difference of recon vs target.
    # This term is zero for "render the temporal mean" (Δ_t target ≠ 0,
    # Δ_t recon = 0 → big penalty). Forces the model to capture per-frame
    # variation, not just per-window means.
    if target.shape[2] >= 2:
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        recon_diff = recon[:, :, 1:] - recon[:, :, :-1]
        diff_loss = F.mse_loss(recon_diff, target_diff)
    else:
        diff_loss = target.new_zeros(())

    # α-weighted smoothness on z_dyn: penalize 2nd-diff only where the slot
    # is "alive" (α near 1). A slot that's not yet born has α≈0, so its
    # z_dyn movements don't contribute — they're effectively unconstrained.
    if z_dyn.shape[1] >= 3:
        d2 = z_dyn[:, 2:] - 2.0 * z_dyn[:, 1:-1] + z_dyn[:, :-2]    # (B, T-2, K, d_dyn)
        a_mask = alpha[:, 2:].unsqueeze(-1)                          # (B, T-2, K, 1)
        smooth_num = (d2.pow(2) * a_mask).sum()
        smooth_den = (a_mask.sum() * d2.shape[-1]).clamp_min(1.0)
        smooth_loss = smooth_num / smooth_den
    else:
        smooth_loss = z_dyn.new_zeros(())

    # Entropy of the per-slot birth-time softmax (over T+1 categories: T birth
    # times + 1 "never born"). Low entropy = peaked = clean event localization.
    full_probs = torch.cat([event_prob, null_prob.unsqueeze(1)], dim=1)  # (B, T+1, K)
    eps = 1e-8
    entropy_per_slot = -(full_probs * (full_probs + eps).log()).sum(dim=1)  # (B, K)
    entropy_loss = entropy_per_slot.mean()

    # VICReg-lite on z_static
    flat = z_static.reshape(-1, z_static.shape[-1])
    flat_centered = flat - flat.mean(dim=0, keepdim=True)
    var_per_dim = torch.sqrt(flat_centered.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - var_per_dim).mean()
    cov = (flat_centered.T @ flat_centered) / max(flat.shape[0] - 1, 1)
    d = cov.shape[0]
    cov_off = cov - cov.diag().diag()
    cov_loss = cov_off.pow(2).sum() / d
    vic_loss = var_loss + cov_loss

    # α-sparsity: uniform L1 on α (legacy, default off).
    alpha_sparse_loss = alpha.mean()

    # Rate-distortion penalty on per-slot bandwidth — the principled fix for
    # event localization. A slot's "cost" at frame t is α[t,k] × ‖z_dyn[t,k]‖².
    # This is non-zero ONLY when a slot is alive AND has non-trivial content.
    # Combined with α-weighted smoothness, this favors α=0 before entry +
    # α=1 after over the degenerate α=1 ∀t with z_dyn rendering invisibility.
    # The rationale: in the degenerate solution, z_dyn at "invisible" frames
    # must be near-zero (to render absence), but the smoothness penalty
    # disallows the sharp jump from near-zero to real content at the entry
    # frame. The α-based gating avoids this by zeroing the slot entirely.
    z_dyn_norm_sq = z_dyn.pow(2).mean(dim=-1)                                # (B, T, K)
    slot_bw_loss = (alpha * z_dyn_norm_sq).mean()

    total = (recon_loss
             + lambda_diff * diff_loss
             + lambda_smooth * smooth_loss
             + lambda_entropy * entropy_loss
             + lambda_vicreg * vic_loss
             + lambda_alpha_sparse * alpha_sparse_loss
             + lambda_slot_bw * slot_bw_loss)

    return {
        "total": total,
        "recon": recon_loss.detach(),
        "diff": diff_loss.detach(),
        "smooth": smooth_loss.detach(),
        "entropy": entropy_loss.detach(),
        "var": var_loss.detach(),
        "cov": cov_loss.detach(),
        "alpha_sparse": alpha_sparse_loss.detach(),
        "slot_bw": slot_bw_loss.detach(),
        "mean_null": null_prob.mean().detach(),
        "mean_alpha_last": alpha[:, -1].mean().detach(),
    }


__all__ = [
    "TrajectoryEncoder",
    "TrajectoryDecoder",
    "DynPredictor",
    "fill_z_dyn",
    "event_to_alpha",
    "trajectory_loss",
    "slot_contrast_loss",
    "event_from_predict_error",
]
