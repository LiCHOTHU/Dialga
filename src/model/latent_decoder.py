"""LatentDecoder — per-frame Wan-latent decoder.

v5.1.1 semantics: (z_static (B, D_s), z_dyn (B, T, D_d)) -> (B, C, T, H, W).

Each frame t is decoded from (z_static, z_dyn[t]) independently — no
temporal mixing inside the decoder. Per-frame z_dyn is required to make
per-frame reconstruction tractable (v5.1's time-collapsed z_dyn plateaued
at L_recon ~0.02 on a 20-vid overfit because one static-in-time vector
cannot encode 9 distinct frame contents).

Linear lift per frame, one refining 2D conv, 1x1 zero-init projection back
to latent_ch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentDecoder(nn.Module):
    def __init__(
        self,
        latent_ch: int = 48,
        d_static: int = 32,
        d_dyn: int = 16,
        hidden_ch: int = 64,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.hidden_ch = int(hidden_ch)
        self.chunk_size_lat = int(chunk_size_lat)
        self.spatial_size = int(spatial_size)

        in_dim = self.d_static + self.d_dyn
        spatial = self.spatial_size * self.spatial_size
        self.in_proj = nn.Linear(in_dim, hidden_ch * spatial)

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GroupNorm(n_groups, hidden_ch),
            nn.SiLU(),
        )
        self.out_proj = nn.Conv2d(hidden_ch, latent_ch, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_static: torch.Tensor, z_dyn: torch.Tensor) -> torch.Tensor:
        """
        z_static : (B, D_s)
        z_dyn    : (B, T_lat, D_d)
        Returns:
            latent : (B, latent_ch, T_lat, H, W)
        """
        if z_static.dim() != 2:
            raise ValueError(f"z_static must be (B, D_s), got {tuple(z_static.shape)}")
        if z_dyn.dim() != 3:
            raise ValueError(f"z_dyn must be (B, T, D_d), got {tuple(z_dyn.shape)}")
        B, T, D_d = z_dyn.shape
        if D_d != self.d_dyn:
            raise ValueError(f"z_dyn last dim {D_d} != d_dyn {self.d_dyn}")

        z_s_exp = z_static.unsqueeze(1).expand(B, T, -1)               # (B, T, D_s)
        x = torch.cat([z_s_exp, z_dyn], dim=-1)                        # (B, T, D_s+D_d)
        x = x.reshape(B * T, -1)                                       # (BT, D_s+D_d)
        h = self.in_proj(x)                                            # (BT, hidden*H*W)
        h = h.reshape(B * T, self.hidden_ch, self.spatial_size, self.spatial_size)
        h = self.refine(h)
        out = self.out_proj(h)                                         # (BT, C, H, W)
        out = out.reshape(B, T, self.latent_ch,
                          self.spatial_size, self.spatial_size)
        return out.permute(0, 2, 1, 3, 4).contiguous()                 # (B, C, T, H, W)


__all__ = ["LatentDecoder"]
