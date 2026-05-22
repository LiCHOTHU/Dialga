"""LatentEncoder3D — chunk-wise encoder over one Wan-latent chunk.

v5.1.1 semantics: ONE chunk in -> z_static (time-collapsed) + z_dyn (per-frame).

The 20-vid overfit at v5.1 (time-collapsed z_dyn) plateaued at L_recon~0.019
even with 4x capacity, because the decoder must produce 9 distinct frames
from a single static-in-time z_dyn. Per-frame z_dyn restores temporal
information for reconstruction while keeping z_static global (the identity
channel that InfoNCE pulls together across chunks).

Two independent Conv3d trunks (no shared backbone) so the encoder cannot
trivially route identity into both heads via the same hidden features.

Input :  (B, C=48, T_lat=9, H=8, W=8)
Output:  dict with
            z_static : (B, D_s=32)         time-pooled
            z_dyn    : (B, T_lat, D_d=16)  per-frame (H,W pooled only)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv3d_trunk(in_ch: int, hidden_ch: int, n_groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, hidden_ch, kernel_size=3, padding=1),
        nn.GroupNorm(n_groups, hidden_ch),
        nn.SiLU(),
        nn.Conv3d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
        nn.GroupNorm(n_groups, hidden_ch),
        nn.SiLU(),
    )


class LatentEncoder3D(nn.Module):
    def __init__(
        self,
        latent_ch: int = 48,
        hidden_ch: int = 32,
        d_static: int = 32,
        d_dyn: int = 16,
        n_groups: int = 8,
        shared_trunk: bool = False,
    ):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.hidden_ch = int(hidden_ch)
        self.d_static = int(d_static)
        self.d_dyn = int(d_dyn)
        self.shared_trunk = bool(shared_trunk)

        self.trunk_static = _conv3d_trunk(latent_ch, hidden_ch, n_groups)
        if self.shared_trunk:
            # Both heads pool from the same trunk's hidden tensor (ablation 2).
            self.trunk_dyn = self.trunk_static
        else:
            self.trunk_dyn = _conv3d_trunk(latent_ch, hidden_ch, n_groups)

        self.head_static = nn.Linear(hidden_ch, d_static)
        self.head_dyn = nn.Linear(hidden_ch, d_dyn)

    def forward(self, latent_chunk: torch.Tensor) -> dict:
        """
        latent_chunk : (B, C, T_lat, H, W)
        Returns dict with
            z_static : (B, D_s)
            z_dyn    : (B, T_lat, D_d)   per-frame
        """
        if latent_chunk.dim() != 5:
            raise ValueError(f"expected 5D (B,C,T,H,W), got {tuple(latent_chunk.shape)}")

        if self.shared_trunk:
            # Compute the shared trunk output once, pool differently for each head.
            h_shared = self.trunk_static(latent_chunk)               # (B, hidden_ch, T, H, W)
            h_s = h_shared.mean(dim=(2, 3, 4))                       # (B, hidden_ch)
            h_d_raw = h_shared
        else:
            h_s = self.trunk_static(latent_chunk).mean(dim=(2, 3, 4))
            h_d_raw = self.trunk_dyn(latent_chunk)
        z_static = self.head_static(h_s)                             # (B, D_s)
        h_d = h_d_raw.mean(dim=(3, 4)).permute(0, 2, 1).contiguous() # (B, T, hidden_ch)
        z_dyn = self.head_dyn(h_d)                                   # (B, T, D_d)
        return {"z_static": z_static, "z_dyn": z_dyn}


__all__ = ["LatentEncoder3D"]
