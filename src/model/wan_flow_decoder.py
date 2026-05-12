"""Flow-matching decoder targeting Wan 2.2 VAE latents.

Latent shape (Wan2.2-TI2V-5B VAE): (B, 48, T_lat=2, 8, 8) for a 6-frame
128x128 input window. Tiny — 128 spatial-temporal tokens — so a small DiT
is plenty.

Forward signature
-----------------
    v_pred = decoder(x_t, t, cond)
where
    x_t : (B, 48, T_lat, 8, 8)        normalized noisy latent
    t   : (B,)                        flow time in [0, 1]
    cond: dict with
        positions  : (B, T, K, 2)     per-frame slot positions
        velocities : (B, T, K, 2)     per-frame slot velocities
        z_static   : (B, K, d_s)      time-invariant per-slot identity
        visibility : (B, T, K)        per-frame visibility mask
        i0_latent  : (B, 48, 1, 8, 8) frozen-VAE encoding of the first frame
                                       (or None if not used)

Conditioning is consumed via cross-attention from latent tokens onto a bank
of conditioning tokens (slot tokens + optional I_0 latent tokens).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# small utilities
# -------------------------------------------------------------------------

def sinusoidal_time_embedding(t, dim):
    """t: (B,) in [0,1]; returns (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    args = t[:, None] * freqs[None, :] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def fourier_pos_features(x, num_freqs=4):
    """x: (..., D); returns (..., D * 2 * num_freqs) sinusoidal features."""
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi
    args = x.unsqueeze(-1) * freqs                                # (..., D, F)
    args = args.flatten(-2)                                       # (..., D*F)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# -------------------------------------------------------------------------
# slot conditioning encoder
# -------------------------------------------------------------------------

class SlotCondEncoder(nn.Module):
    """Per-slot, per-frame token bank for cross-attention conditioning."""

    def __init__(self, d_static, d_model, window_length, max_objects,
                 num_pos_freqs=4):
        super().__init__()
        self.window_length = int(window_length)
        self.max_objects = int(max_objects)
        self.num_pos_freqs = int(num_pos_freqs)

        per_pos_dim = 2 * 2 * num_pos_freqs                       # sincos(pos) + sincos(vel) below
        feat_in = (per_pos_dim                                    # positions
                   + per_pos_dim                                  # velocities
                   + int(d_static)                                # z_static
                   + 1)                                            # visibility
        self.token_mlp = nn.Sequential(
            nn.Linear(feat_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.frame_emb = nn.Embedding(self.window_length, d_model)
        self.slot_emb = nn.Embedding(self.max_objects, d_model)

    def forward(self, positions, velocities, z_static, visibility):
        """
        positions  : (B, T, K, 2)
        velocities : (B, T, K, 2)
        z_static   : (B, K, d_s)
        visibility : (B, T, K)
        Returns
            tokens : (B, T*K, d_model)
            mask   : (B, T*K) bool — True where visible
        """
        B, T, K, _ = positions.shape
        pos_feat = fourier_pos_features(positions, self.num_pos_freqs)        # (B, T, K, F)
        vel_feat = fourier_pos_features(velocities, self.num_pos_freqs)
        zs = z_static.unsqueeze(1).expand(-1, T, -1, -1)                       # (B, T, K, d_s)
        feats = torch.cat([pos_feat, vel_feat, zs, visibility.unsqueeze(-1)], dim=-1)
        tokens = self.token_mlp(feats)                                         # (B, T, K, d_model)

        frame_ids = torch.arange(T, device=positions.device)
        slot_ids = torch.arange(K, device=positions.device)
        tokens = tokens + self.frame_emb(frame_ids)[None, :, None, :]
        tokens = tokens + self.slot_emb(slot_ids)[None, None, :, :]

        tokens = tokens.reshape(B, T * K, -1)
        mask = visibility.reshape(B, T * K) > 0.5
        return tokens, mask


# -------------------------------------------------------------------------
# DiT-style transformer block with cross-attention
# -------------------------------------------------------------------------

class AdaLN(nn.Module):
    """AdaLN-Zero: time-conditioned scale, shift, gate."""

    def __init__(self, d_model, t_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(t_dim, 3 * d_model)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, t_emb):
        scale, shift, gate = self.modulation(t_emb).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, ctx, ctx_mask):
        B, N, D = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask[:, None, None, :].expand(B, self.h, N, M)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class DiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, t_dim, mlp_ratio=4):
        super().__init__()
        self.adaln1 = AdaLN(d_model, t_dim)
        self.self_attn = SelfAttention(d_model, n_heads)
        self.adaln2 = AdaLN(d_model, t_dim)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.adaln3 = AdaLN(d_model, t_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, x, t_emb, ctx, ctx_mask):
        h, gate = self.adaln1(x, t_emb)
        x = x + gate * self.self_attn(h)
        h, gate = self.adaln2(x, t_emb)
        x = x + gate * self.cross_attn(h, ctx, ctx_mask)
        h, gate = self.adaln3(x, t_emb)
        x = x + gate * self.mlp(h)
        return x


# -------------------------------------------------------------------------
# main module
# -------------------------------------------------------------------------

class WanLatentFlowDecoder(nn.Module):
    """DiT-style flow-matching velocity predictor for Wan2.2 latent space."""

    def __init__(
        self,
        latent_channels=48,
        latent_grid=8,
        latent_T=2,
        d_model=384,
        n_heads=6,
        n_blocks=6,
        t_dim=384,
        d_static=16,
        max_objects=6,
        window_length=6,
        use_i0=True,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_grid = int(latent_grid)
        self.latent_T = int(latent_T)
        self.use_i0 = bool(use_i0)
        N = self.latent_T * self.latent_grid * self.latent_grid

        self.patch_in = nn.Linear(latent_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, N, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.slot_enc = SlotCondEncoder(
            d_static=d_static, d_model=d_model,
            window_length=window_length, max_objects=max_objects,
        )

        if self.use_i0:
            self.ref_proj = nn.Linear(latent_channels, d_model)
            ref_N = 1 * self.latent_grid * self.latent_grid
            self.ref_pos_emb = nn.Parameter(torch.zeros(1, ref_N, d_model))
            nn.init.normal_(self.ref_pos_emb, std=0.02)
            self.ref_role_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.ref_role_emb, std=0.02)
            self.slot_role_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.slot_role_emb, std=0.02)

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.t_dim = t_dim

        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, t_dim) for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.t_modulation_out = nn.Linear(t_dim, 2 * d_model)
        nn.init.zeros_(self.t_modulation_out.weight)
        nn.init.zeros_(self.t_modulation_out.bias)
        self.patch_out = nn.Linear(d_model, latent_channels)
        nn.init.zeros_(self.patch_out.weight)
        nn.init.zeros_(self.patch_out.bias)

    def _patchify(self, x):
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)

    def _unpatchify(self, x, T, H, W):
        B, _, C = x.shape
        return x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x_t, t, positions, velocities, z_static, visibility, i0_latent=None):
        """
        x_t        : (B, C, T_lat, H, W)
        t          : (B,) in [0, 1]
        positions  : (B, T, K, 2)
        velocities : (B, T, K, 2)
        z_static   : (B, K, d_s)
        visibility : (B, T, K)
        i0_latent  : (B, C, 1, H, W) or None
        Returns velocity tensor with the same shape as x_t.
        """
        B, C, T_lat, H, W = x_t.shape

        # latent tokens
        tokens = self.patch_in(self._patchify(x_t))                   # (B, N, d)
        tokens = tokens + self.pos_emb

        # time embedding
        t_emb = sinusoidal_time_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)

        # conditioning tokens
        slot_tokens, slot_mask = self.slot_enc(positions, velocities, z_static, visibility)
        if self.use_i0:
            slot_tokens = slot_tokens + self.slot_role_emb
            ctx_list = [slot_tokens]
            mask_list = [slot_mask]
            if i0_latent is not None:
                ref = self.ref_proj(self._patchify(i0_latent))         # (B, ref_N, d)
                ref = ref + self.ref_pos_emb + self.ref_role_emb
                ctx_list.append(ref)
                ref_mask = torch.ones(B, ref.shape[1], dtype=torch.bool, device=x_t.device)
                mask_list.append(ref_mask)
            ctx = torch.cat(ctx_list, dim=1)
            ctx_mask = torch.cat(mask_list, dim=1)
        else:
            ctx = slot_tokens
            ctx_mask = slot_mask

        # transformer
        for block in self.blocks:
            tokens = block(tokens, t_emb, ctx, ctx_mask)

        # final AdaLN + project back to latent channels
        scale, shift = self.t_modulation_out(t_emb).chunk(2, dim=-1)
        tokens = self.norm_out(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        out = self.patch_out(tokens)
        return self._unpatchify(out, T_lat, H, W)


__all__ = ["WanLatentFlowDecoder"]
