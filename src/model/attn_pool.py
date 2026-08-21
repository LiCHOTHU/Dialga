"""AttentionPool — multi-query attention pooling over spatial cells.

Replaces the global mean-pool in LatentEncoder3D. The internals inspection
(2026-06-12) showed the mean-pool is the model's chokepoint: it averages 64
spatial cells per dyn frame (576 for static), and although the conv trunk
produces object cells that are clearly distinct from background cells
(feature distance 2.6, top-10% of cells hold only ~20% of the norm), the mean
dilutes any 2-3-cell object ~25:1. Effective dimensionality collapses to
~15/96 (static) and ~10/96 (dyn).

Attention pooling lets M learned query vectors *select* which cells to read
via softmax attention, so a query can latch onto one object's cells instead of
being forced to average over the whole frame. M is set near the CLEVRER object
count (8). Output is the concatenation of the M pooled heads, projected back to
`out_dim` (= hidden_ch), so the existing head Linear keeps the same interface.

Standard scaled-dot-product cross-attention (Set-Transformer PMA / Perceiver
style): queries are learned parameters; keys/values are the cells.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        n_queries: int = 8,
        n_heads: int = 4,
        slot_mode: bool = False,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = int(dim)
        self.out_dim = int(out_dim if out_dim is not None else dim)
        self.n_queries = int(n_queries)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        # v5.3: slot_mode keeps the per-query (slot) axis instead of fusing the
        # M heads into one vector. Output is then (B, M, out_dim) — the latent
        # retains object-separated structure rather than re-globalizing.
        self.slot_mode = bool(slot_mode)

        # Learned query bank (the "slots" that read the cells).
        self.queries = nn.Parameter(torch.randn(self.n_queries, self.dim) * 0.02)
        self.to_q = nn.Linear(self.dim, self.dim)
        self.to_k = nn.Linear(self.dim, self.dim)
        self.to_v = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        if self.slot_mode:
            # Per-slot projection (shared across slots), preserves the M axis.
            self.slot_proj = nn.Linear(self.dim, self.out_dim)
        else:
            # Fuse the M pooled heads back to out_dim.
            self.proj = nn.Linear(self.n_queries * self.dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, dim) — N spatial cells.
        Returns (B, out_dim), or (B, n_queries, out_dim) if slot_mode.
        """
        if x.dim() != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"expected (B, N, {self.dim}), got {tuple(x.shape)}")
        B, N, _ = x.shape
        x = self.norm(x)
        q = self.to_q(self.queries).unsqueeze(0).expand(B, -1, -1)       # (B, M, dim)
        k = self.to_k(x)                                                  # (B, N, dim)
        v = self.to_v(x)                                                  # (B, N, dim)

        def split(t, L):
            return t.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        qh = split(q, self.n_queries)                                    # (B, h, M, hd)
        kh = split(k, N)                                                 # (B, h, N, hd)
        vh = split(v, N)
        attn = (qh @ kh.transpose(-2, -1)) / (self.head_dim ** 0.5)      # (B, h, M, N)
        if self.slot_mode:
            # Slot Attention competition (Locatello): softmax over the SLOT axis
            # so each cell is divided AMONG the M slots (they compete for cells),
            # then a per-slot weighted mean over cells. This is the inductive bias
            # the PMA-style softmax(dim=-1) lacked — without it the M queries read
            # the scene independently and overlap/memorize (val attrs CE blew up).
            attn = attn.softmax(dim=2)                                   # compete over M
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)        # per-slot mean
        else:
            attn = attn.softmax(dim=-1)
        out = attn @ vh                                                  # (B, h, M, hd)
        out = out.transpose(1, 2).reshape(B, self.n_queries, self.dim)   # (B, M, dim)
        if self.slot_mode:
            return self.slot_proj(out)                                   # (B, M, out_dim)
        out = out.reshape(B, self.n_queries * self.dim)                  # (B, M*dim)
        return self.proj(out)                                            # (B, out_dim)


__all__ = ["AttentionPool"]
