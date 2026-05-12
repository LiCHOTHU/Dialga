"""
Slot-state encoders + pixel decoder + collision impulse.

Two encoder variants share the same downstream:
  - SlotQueryEncoder    : K query tokens cross-attend to image patches → (K, 2)
  - LatentSIGRegEncoder : flat latent + SIGReg regularizer + projection → (K, 2)

Both produce a (B, K, 2) per-object position state. Downstream:
  - CollisionImpulse       : physics-parameterized 2-body collision impulse (Stage 2)
  - SlotPixelDecoder       : (positions, attrs, visibility) → image (Stage 2)

The masking convention everywhere:
  α (alpha) ∈ {0, 1}^K : per-slot visibility (1 = inside camera view).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------- encoders


class _SharedPatchBackbone(nn.Module):
    """Conv2d patch embed + Transformer encoder. Used by both encoder variants."""

    def __init__(self, image_size, patch_size, in_channels, embed_dim,
                 depth, num_heads, mlp_ratio):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = image_size // patch_size
        self.embed_dim = int(embed_dim)

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(embed_dim * float(mlp_ratio)),
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.out_norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, N, D)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.encoder(tokens)
        return self.out_norm(tokens)


class SlotQueryEncoder(nn.Module):
    """
    image (B, 3, H, W) + slot attrs (B, K, A)
        → positions (B, K, 2), z_static (B, K, d_static)

    K learnable query tokens (one per sorted-object_id slot) cross-attend to
    image patch tokens. Each query is offset by an attribute embedding so that
    the same slot index can specialize to whatever object id sits there.

    Two heads on the cross-attended slot embedding:
      - slot_mlp     → 2D position q
      - z_static_head → per-frame identity latent z_static. Downstream we
        pool over time to one vector per (video, slot) and feed it to
        AccelNet + decoder in place of GT attrs.

    GT attrs still bias the slot queries here so slot↔object alignment is
    preserved; they are *not* propagated past the encoder.
    """

    def __init__(self, image_size=128, patch_size=16, in_channels=3,
                 embed_dim=192, depth=4, num_heads=6, mlp_ratio=4.0,
                 max_objects=8, attr_dim=13, num_state_dims=2, d_static=16):
        super().__init__()
        self.max_objects = int(max_objects)
        self.num_state_dims = int(num_state_dims)
        self.d_static = int(d_static)

        self.backbone = _SharedPatchBackbone(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )

        self.slot_queries = nn.Parameter(torch.zeros(1, max_objects, embed_dim))
        self.attr_proj = nn.Linear(attr_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=int(num_heads), batch_first=True
        )
        self.cross_norm_q = nn.LayerNorm(embed_dim)
        self.cross_norm_kv = nn.LayerNorm(embed_dim)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_state_dims),
        )
        self.z_static_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.d_static),
        )

        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        nn.init.zeros_(self.slot_mlp[-1].weight)
        nn.init.zeros_(self.slot_mlp[-1].bias)

    def forward(self, image, attrs):
        """image: (B, 3, H, W), attrs: (B, K, A)
        → positions (B, K, num_state_dims), z_static (B, K, d_static)
        """
        B = image.shape[0]
        kv = self.backbone(image)                              # (B, N, D)
        q = self.slot_queries.expand(B, -1, -1) + self.attr_proj(attrs)  # (B, K, D)
        q_n = self.cross_norm_q(q)
        kv_n = self.cross_norm_kv(kv)
        attended, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        out = q + attended                                     # residual
        positions = self.slot_mlp(out)                         # (B, K, num_state_dims)
        z_static = self.z_static_head(out)                     # (B, K, d_static)
        return positions, z_static


class LatentSIGRegEncoder(nn.Module):
    """
    image (B, 3, H, W) + slot attrs (B, K, A) → positions (B, K, 2)

    Shared backbone → flat latent (mean-pooled tokens, like CLS) → projection
    head that maps the latent + per-slot attrs to per-object positions.

    The flat latent is exposed via .latent(image) so SIGReg can be applied to it
    during training. The latent itself is unstructured; per-object structure
    only comes from the projection head + attribute conditioning + GT supervision.
    """

    def __init__(self, image_size=128, patch_size=16, in_channels=3,
                 embed_dim=192, depth=4, num_heads=6, mlp_ratio=4.0,
                 max_objects=8, attr_dim=13, num_state_dims=2,
                 latent_dim=128, d_static=16):
        super().__init__()
        self.max_objects = int(max_objects)
        self.num_state_dims = int(num_state_dims)
        self.latent_dim = int(latent_dim)
        self.d_static = int(d_static)

        self.backbone = _SharedPatchBackbone(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )
        self.latent_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, latent_dim)
        )
        self.attr_embed = nn.Linear(attr_dim, latent_dim)
        self.proj_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, num_state_dims),
        )
        self.z_static_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.d_static),
        )
        nn.init.zeros_(self.proj_head[-1].weight)
        nn.init.zeros_(self.proj_head[-1].bias)

    def latent(self, image):
        """(B, 3, H, W) → (B, latent_dim) flat latent (for SIGReg)."""
        tokens = self.backbone(image)                          # (B, N, D)
        pooled = tokens.mean(dim=1)                            # (B, D)
        return self.latent_head(pooled)                        # (B, latent_dim)

    def forward(self, image, attrs):
        """(B, 3, H, W), (B, K, A) → positions (B, K, 2), z_static (B, K, d_static)."""
        z = self.latent(image)                                 # (B, L)
        per_slot = z.unsqueeze(1) + self.attr_embed(attrs)     # (B, K, L)
        positions = self.proj_head(per_slot)                   # (B, K, num_state_dims)
        z_static = self.z_static_head(per_slot)                # (B, K, d_static)
        return positions, z_static


# -------------------------------------------------------------------- decoder


class SlotPixelDecoder(nn.Module):
    """
    Slot → pixel decoder.

    Each slot k produces a per-slot embedding from (q_k, attr_k, α_k). All slot
    embeddings are broadcast onto a 2D spatial grid (with positional encoding)
    and passed through a small CNN that produces per-slot RGB + alpha images.
    Composition is via softmax-over-slots on the alpha logits, masked by the
    per-slot visibility α_k.

    This is a slot-attention-style broadcast decoder, but with explicit
    per-slot input from the predicted positions and attributes (no Slot
    Attention competition needed since slot identity is fixed).
    """

    def __init__(self, num_state_dims=2, attr_dim=13, max_objects=8,
                 image_size=128, hidden=64, slot_embed=128, num_blocks=3,
                 grid_size=16):
        super().__init__()
        self.D = int(num_state_dims)
        self.image_size = int(image_size)
        self.grid_size = int(grid_size)
        self.max_objects = int(max_objects)

        # Slot conditioning
        self.slot_mlp = nn.Sequential(
            nn.Linear(self.D + attr_dim + 1, slot_embed), nn.SiLU(),
            nn.Linear(slot_embed, slot_embed),
        )

        # Coordinate grid (broadcast features to spatial positions)
        ys = torch.linspace(-1.0, 1.0, self.grid_size)
        xs = torch.linspace(-1.0, 1.0, self.grid_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coord = torch.stack([gx, gy], dim=0).unsqueeze(0)         # (1, 2, G, G)
        self.register_buffer("coord", coord)

        # slot embed broadcast + absolute coord channels (2) + q-relative coord
        # channels (2). The q-relative channels are what force the decoder to
        # be identifiable in q: shifting q shifts the slot-local coord frame,
        # so the CNN's output pixels follow.
        in_ch = slot_embed + 4
        layers = []
        c = in_ch
        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(c, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.SiLU(),
            ]
            c = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, 4, kernel_size=1)           # (rgb + alpha-logit)
        self.background = nn.Parameter(torch.zeros(3, image_size, image_size))

        upsample_factor = image_size // self.grid_size
        if image_size % self.grid_size != 0:
            raise ValueError("image_size must be divisible by grid_size")
        self.upsample_factor = upsample_factor

    def forward(self, positions, attrs, alpha):
        """
        positions: (B, K, D), attrs: (B, K, A), alpha: (B, K)
        returns:   (B, 3, H, W)

        Identifiability-in-q: in addition to the absolute coord grid, we also
        broadcast a *q-relative* coord channel [gx - q_x, gy - q_y]. This shifts
        the CNN's input frame to be slot-centered, so the rendered slot's
        spatial location is *forced* to track q. Without this, the absolute
        coord channels alone let the decoder ignore q (slot_emb can encode
        position internally), which yields the degenerate constant-velocity
        encoder solution.
        """
        B, K = alpha.shape
        slot_in = torch.cat([positions, attrs, alpha.unsqueeze(-1)], dim=-1)  # (B, K, D+A+1)
        slot_emb = self.slot_mlp(slot_in)                                    # (B, K, E)

        # Broadcast slot embedding onto the spatial grid
        flat = slot_emb.view(B * K, -1, 1, 1).expand(-1, -1, self.grid_size, self.grid_size)
        coord = self.coord.expand(B * K, -1, -1, -1)                          # (B*K, 2, G, G)
        # q-relative coords: subtract slot's predicted position (clamped to the
        # decoder's coord range so far-off positions don't blow up the CNN).
        q_clamped = positions[..., :2].clamp(-1.5, 1.5)                       # (B, K, 2)
        q_grid = q_clamped.view(B * K, 2, 1, 1).expand(-1, -1, self.grid_size, self.grid_size)
        coord_rel = coord - q_grid                                            # (B*K, 2, G, G)
        x = torch.cat([flat, coord, coord_rel], dim=1)                        # (B*K, E+4, G, G)
        x = self.conv(x)
        x = self.head(x)                                                     # (B*K, 4, G, G)

        # Upsample to image resolution
        x = F.interpolate(x, scale_factor=self.upsample_factor, mode="bilinear", align_corners=False)
        x = x.view(B, K, 4, self.image_size, self.image_size)
        rgb = x[:, :, :3]
        alpha_logit = x[:, :, 3]                                             # (B, K, H, W)

        # Mask invisible slots out by setting their alpha logit very negative
        invis = (alpha < 0.5).view(B, K, 1, 1)
        alpha_logit = alpha_logit.masked_fill(invis, -1e4)

        # softmax-over-slots → composition weights
        weights = F.softmax(alpha_logit, dim=1).unsqueeze(2)                 # (B, K, 1, H, W)
        composed = (weights * torch.tanh(rgb)).sum(dim=1)                    # (B, 3, H, W)

        # Where all slots are masked out (no objects), softmax becomes uniform
        # over zero contribution; add a learnable background to fill.
        any_visible = (alpha.sum(dim=-1, keepdim=True) > 0).float().view(B, 1, 1, 1)
        bg = self.background.unsqueeze(0).tanh()
        return any_visible * composed + (1.0 - any_visible) * bg


# -------------------------------------------------------------------- collision impulse


class CollisionImpulse(nn.Module):
    """
    Physics-parameterized 2-body collision impulse.

    Given pre-collision positions q_i, q_j and velocities v_i, v_j (and per-slot
    attributes), apply the standard inelastic-collision-with-restitution update
    along the line of centers:

        n̂ = (q_j - q_i) / ||q_j - q_i||
        v_rel_n = (v_i - v_j) · n̂                           (positive ⇒ approaching)
        e = sigmoid(MLP(attrs_i, attrs_j))                  ∈ (0, 1)
        Δp = -(1 + e) · (m_i m_j / (m_i + m_j)) · v_rel_n   (along n̂)
        v_i_post = v_i + Δp · n̂ / m_i
        v_j_post = v_j - Δp · n̂ / m_j

    Masses come from the Lagrangian (kept consistent with the kinetic term).

    Only impulses with v_rel_n > 0 (objects approaching) are applied — separating
    pairs are passed through unchanged. The restitution head is symmetric in
    (i, j) by averaging both orderings.
    """

    def __init__(self, attr_dim, hidden=64, init_e_logit=0.0):
        super().__init__()
        self.attr_dim = int(attr_dim)
        self.e_head = nn.Sequential(
            nn.Linear(2 * attr_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.e_head[-1].weight)
            self.e_head[-1].bias.fill_(float(init_e_logit))

    def restitution(self, attrs_i, attrs_j):
        """Symmetric restitution e ∈ (0, 1). attrs_*: (..., A) → (...,)"""
        x_fwd = torch.cat([attrs_i, attrs_j], dim=-1)
        x_rev = torch.cat([attrs_j, attrs_i], dim=-1)
        logit = 0.5 * (self.e_head(x_fwd) + self.e_head(x_rev)).squeeze(-1)
        return torch.sigmoid(logit)

    def forward(self, q_i, q_j, v_i, v_j, attrs_i, attrs_j, mass_i, mass_j,
                eps=1e-6):
        """
        All inputs shaped (..., D) for q,v and (..., A) for attrs and (...,) for mass.
        Returns (v_i_post, v_j_post) of shape (..., D).
        """
        delta = q_j - q_i                                            # (..., D)
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(eps)
        n_hat = delta / dist                                          # (..., D)
        v_rel = v_i - v_j                                             # (..., D)
        v_rel_n = (v_rel * n_hat).sum(dim=-1, keepdim=True)           # (..., 1)

        e = self.restitution(attrs_i, attrs_j).unsqueeze(-1)          # (..., 1)
        m_eff = (mass_i * mass_j) / (mass_i + mass_j).clamp_min(eps)  # (...)
        m_eff = m_eff.unsqueeze(-1)                                   # (..., 1)

        # Only resolve approaching pairs (v_rel_n > 0). Otherwise pass through.
        approaching = (v_rel_n > 0).float()
        impulse_mag = -(1.0 + e) * m_eff * v_rel_n * approaching      # (..., 1)
        impulse = impulse_mag * n_hat                                  # (..., D)

        v_i_post = v_i + impulse / mass_i.unsqueeze(-1)
        v_j_post = v_j - impulse / mass_j.unsqueeze(-1)
        return v_i_post, v_j_post


def apply_collision_impulse_to_positions(impulse_module, dynamics,
                                         q_prev, q_curr, attrs, collision_pairs):
    """
    Helper for rollout: convert (q_prev, q_curr) into a velocity, apply impulse
    for each (slot_i, slot_j) pair, return a corrected q_prev_eq such that
    `q_curr - q_prev_eq` equals the post-collision velocity for each pair's
    slots while leaving other slots untouched.

    This lets the next dynamics step "see" post-collision velocity via the
    standard q_prev → q_curr → q_next formulation.

        q_prev, q_curr: (B, K, D)
        attrs:          (B, K, A)
        collision_pairs: list (length B) of list[(t, i, j)]; only pairs with
                         t == 0 are applied here (caller filters to current step).

    `dynamics` is the module that exposes `_mass(attrs) → (B, K)` (AccelNet).

    Returns q_prev_corrected: (B, K, D).
    """
    B, K, D = q_curr.shape
    v = q_curr - q_prev                                              # (B, K, D)
    masses = dynamics._mass(attrs)                                   # (B, K)
    out = q_prev.clone()
    for b in range(B):
        for (_t, i, j) in collision_pairs[b]:
            v_i, v_j = v[b, i].unsqueeze(0), v[b, j].unsqueeze(0)
            q_i, q_j = q_curr[b, i].unsqueeze(0), q_curr[b, j].unsqueeze(0)
            a_i, a_j = attrs[b, i].unsqueeze(0), attrs[b, j].unsqueeze(0)
            m_i, m_j = masses[b, i:i+1], masses[b, j:j+1]
            v_i_p, v_j_p = impulse_module(q_i, q_j, v_i, v_j, a_i, a_j, m_i, m_j)
            out[b, i] = q_curr[b, i] - v_i_p.squeeze(0)
            out[b, j] = q_curr[b, j] - v_j_p.squeeze(0)
    return out


__all__ = [
    "SlotQueryEncoder",
    "LatentSIGRegEncoder",
    "CollisionImpulse",
    "SlotPixelDecoder",
    "apply_collision_impulse_to_positions",
]
