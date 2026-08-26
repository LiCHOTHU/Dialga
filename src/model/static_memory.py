"""StaticMemory — build the static scene code from a video, on two levels.

    WITHIN a chunk   collapse the T per-frame static grids into one grid
                     mean   plain temporal average (current DIALGA)
                     median robust average; rejects a mover that is only briefly
                            at a cell (MosaicMem's world-memory trick)
                     sweep  PlaneSweepAggregator: known-pose plane sweep recovers
                            per-cell inverse depth, warps every frame to ONE
                            camera-canonical grid
                     world  WorldMemoryAggregator: same sweep, median accumulation

    ACROSS chunks    carry the grid along the video
                     none   throw it away and recompute (current DIALGA)
                     ema    learnable exponential moving average
                     gru    ConvGRU with learned per-cell read/write gates
                     attn   CUT3R-style: memory as g*g tokens, cross-attends to
                            the new chunk's evidence, gated residual write

Why the second level exists, measured on the CLEVRER Wan cache
(scripts/local/diag_static_memory.py): the per-chunk static estimate DRIFTS along a
video -- rel-MSE 0.083 / 0.154 / 0.187 at chunk lag 1/2/3 on raw latents, and
0.379 / 0.573 / 0.753 for a *learned* per-chunk code. Robust collapse does not fix
it (median 0.0840 vs mean 0.0834): inside a 33-frame chunk an object barely moves,
so a median keeps it; between chunks it has moved. Only accumulation can.

The across-chunk updates are causal -- M_k summarises chunks 0..k -- so the model
runs online at constant cost per chunk, the MUSt3R/CUT3R property.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.camera_pose import PlaneSweepAggregator, WorldMemoryAggregator

UPDATES = ("none", "ema", "gru", "attn")
COLLAPSES = ("mean", "median", "sweep", "world")


class StaticMemory(nn.Module):
    def __init__(self, update: str = "none", collapse: str = "mean",
                 ch: int = 192, grid: int = 4, n_heads: int = 4,
                 d_pose: int = 0, n_frames: int = 9):
        super().__init__()
        if update not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}, got {update!r}")
        if collapse not in COLLAPSES:
            raise ValueError(f"collapse must be one of {COLLAPSES}, got {collapse!r}")
        if collapse in ("sweep", "world") and d_pose <= 0:
            raise ValueError(f"collapse={collapse!r} needs d_pose>0 (known camera)")
        self.update, self.collapse_mode = update, collapse
        self.ch, self.grid, self.d_pose = int(ch), int(grid), int(d_pose)

        if collapse == "sweep":
            self.agg = PlaneSweepAggregator(ch, d_pose, n_frames=n_frames, grid=grid)
        elif collapse == "world":
            self.agg = WorldMemoryAggregator(ch, d_pose, n_frames=n_frames, grid=grid)

        if update == "ema":
            self.alpha = nn.Parameter(torch.zeros(1))       # sigmoid(0)=0.5
        elif update == "gru":
            c = self.ch
            self.conv_z = nn.Conv2d(2 * c, c, 3, padding=1)
            self.conv_r = nn.Conv2d(2 * c, c, 3, padding=1)
            self.conv_h = nn.Conv2d(2 * c, c, 3, padding=1)
            nn.init.zeros_(self.conv_z.bias)
            nn.init.zeros_(self.conv_h.weight)
            nn.init.zeros_(self.conv_h.bias)
        elif update == "attn":
            c = self.ch
            self.norm_m, self.norm_s = nn.LayerNorm(c), nn.LayerNorm(c)
            self.attn = nn.MultiheadAttention(c, n_heads, batch_first=True)
            self.gate, self.out = nn.Linear(2 * c, c), nn.Linear(c, c)
            nn.init.zeros_(self.out.weight)      # M_0 == S_0 exactly at init
            nn.init.zeros_(self.out.bias)
            nn.init.constant_(self.gate.bias, -2.0)

    # ---------------------------------------------------------------- collapse
    def collapse(self, feat: torch.Tensor, pose_emb=None) -> torch.Tensor:
        """feat : (B,T,C,g,g) -> (B,C,g,g)."""
        if self.collapse_mode == "median":
            return feat.median(dim=1).values
        if self.collapse_mode in ("sweep", "world"):
            return self.agg(feat, pose_emb)
        return feat.mean(dim=1)

    # ------------------------------------------------------------------ update
    def forward(self, mem, feat: torch.Tensor, pose_emb=None) -> torch.Tensor:
        s = self.collapse(feat, pose_emb)
        if mem is None or self.update == "none":
            return s
        if self.update == "ema":
            a = torch.sigmoid(self.alpha)
            return (1.0 - a) * mem + a * s
        if self.update == "gru":
            x = torch.cat([mem, s], dim=1)
            z = torch.sigmoid(self.conv_z(x))
            r = torch.sigmoid(self.conv_r(x))
            h = torch.tanh(self.conv_h(torch.cat([r * mem, s], dim=1)))
            return (1.0 - z) * mem + z * h
        B, C, g, _ = mem.shape
        m = mem.flatten(2).transpose(1, 2)
        e = s.flatten(2).transpose(1, 2)
        upd, _ = self.attn(self.norm_m(m), self.norm_s(e), self.norm_s(e),
                           need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat([m, upd], -1)))
        m = m + gate * self.out(upd)
        return m.transpose(1, 2).reshape(B, C, g, g)


__all__ = ["StaticMemory", "UPDATES", "COLLAPSES"]
