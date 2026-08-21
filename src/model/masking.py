"""ObjectRegionMask — saliency-guided latent-cell masking (MAETok, Fork C).

MAETok (arXiv:2502.03444) masks 40-60% of ViT image-patch tokens and trains
auxiliary decoders to predict semantic features (DINOv2) at the masked
positions; that pressure is what makes the latent tokens carry global scene
structure instead of locally copying input.

Our encoder is Conv3d over a Wan-latent chunk (B, 48, 9, 8, 8) — no token
axis — so the port masks SPATIO-TEMPORAL CELLS of the input chunk instead:
a masked cell (t, h, w) has its 48-dim channel vector replaced by one shared
learnable mask vector before the conv trunk sees it.

Saliency guidance: CLEVRER backgrounds are flat, objects are not. Per-cell
saliency = variance across the 48 latent channels. Masked cells are sampled
uniformly from the top `pool_frac` of cells by saliency, so masking
concentrates where objects live (MAETok's "mask where information is")
without being deterministic.

TUBE masking (v5.4, the "gaze" fix). The default per-cell scheme masks over
the full T*H*W volume, which is defeated by TEMPORAL COPYING: a masked cell
(t,h,w) is recoverable from its visible twin (t+-1,h,w), so the aux loss is
satisfiable without the pooled code carrying any structure. With tube=True we
instead select SPATIAL COLUMNS (h,w) by saliency (aggregated over time) and
mask the whole column across ALL t, removing the temporal twin (VideoMAE's
fix). The conv receptive field can still leak from spatial neighbours (h+-1),
but at 8x8 latent resolution adjacent columns are genuinely different content,
so that leak is minor compared with the temporal one this removes.

The masked chunk feeds ONLY the auxiliary semantic-prediction loss; the main
reconstruction/dynamics path always sees the unmasked chunk.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ObjectRegionMask(nn.Module):
    def __init__(self, latent_ch: int = 48, ratio: float = 0.5,
                 pool_frac: float = 0.75, tube: bool = False):
        super().__init__()
        if not 0.0 < ratio <= pool_frac <= 1.0:
            raise ValueError(f"need 0 < ratio <= pool_frac <= 1, got {ratio}, {pool_frac}")
        self.latent_ch = int(latent_ch)
        self.ratio = float(ratio)
        self.pool_frac = float(pool_frac)
        self.tube = bool(tube)
        self.mask_vec = nn.Parameter(torch.zeros(latent_ch))
        nn.init.normal_(self.mask_vec, std=0.02)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        chunk : (B, C, T, H, W)
        Returns:
            masked_chunk : (B, C, T, H, W)  masked cells replaced by mask_vec
            mask         : (B, T, H, W)     bool, True at masked cells
        """
        if chunk.dim() != 5 or chunk.shape[1] != self.latent_ch:
            raise ValueError(f"expected (B,{self.latent_ch},T,H,W), got {tuple(chunk.shape)}")
        B, C, T, H, W = chunk.shape

        with torch.no_grad():
            if self.tube:
                # Select spatial COLUMNS (h,w) by time-aggregated saliency, then
                # mask each chosen column across all t (no temporal twin to copy).
                Ncol = H * W
                n_pool = max(1, math.ceil(self.pool_frac * Ncol))
                n_mask = min(max(1, math.ceil(self.ratio * Ncol)), n_pool)
                sal = chunk.var(dim=1).mean(dim=1).reshape(B, Ncol)     # (B, H*W)
                pool_idx = sal.topk(n_pool, dim=1).indices              # (B, n_pool)
                choice = torch.rand(B, n_pool, device=chunk.device).argsort(dim=1)[:, :n_mask]
                col_idx = pool_idx.gather(1, choice)                    # (B, n_mask)
                col_mask = torch.zeros(B, Ncol, dtype=torch.bool, device=chunk.device)
                col_mask.scatter_(1, col_idx, True)
                mask = col_mask.reshape(B, 1, H, W).expand(B, T, H, W).contiguous()
            else:
                N = T * H * W
                n_pool = max(1, math.ceil(self.pool_frac * N))
                n_mask = min(max(1, math.ceil(self.ratio * N)), n_pool)
                sal = chunk.var(dim=1).reshape(B, N)                    # (B, N)
                pool_idx = sal.topk(n_pool, dim=1).indices              # (B, n_pool)
                # uniform choice of n_mask cells within each sample's saliency pool
                choice = torch.rand(B, n_pool, device=chunk.device).argsort(dim=1)[:, :n_mask]
                mask_idx = pool_idx.gather(1, choice)                   # (B, n_mask)
                mask = torch.zeros(B, N, dtype=torch.bool, device=chunk.device)
                mask.scatter_(1, mask_idx, True)
                mask = mask.reshape(B, T, H, W)

        fill = self.mask_vec.view(1, C, 1, 1, 1).to(chunk.dtype)
        masked = torch.where(mask.unsqueeze(1), fill, chunk)
        return masked, mask


__all__ = ["ObjectRegionMask"]
