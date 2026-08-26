"""VideoStatic — ONE static code for a whole video, projected into each chunk's view.

The memories built so far are causal: M_k summarises chunks 0..k, so z_static still
changes from chunk to chunk and "static" is only ever a tendency. Here a single code
serves every chunk of the video, which makes the word mean something by construction:
if one code has to explain all K chunks, anything that CHANGES cannot live in it, and
the movers are pushed into z_dyn as the residual. No loss term can be gamed into or
out of that -- it is a property of the wiring.

It is also where a real rate win is, as opposed to a redistribution. Per video:

    per-chunk static   K * (d_static + d_dyn)
    video-level        d_static + K * d_dyn        saving -> d_static/(d_static+d_dyn)

which is only worth having if d_static is LARGE. At 96 floats the asymptotic saving is
4% and there is nothing to see; at 768 it is 25%. Measured PCA on the time-constant
target says 96 floats keeps 68.3% of it and 768 keeps 97.1% -- so a big static code is
both affordable (paid once) and useful (it can actually hold the scene). Small static
code + video-level aggregation is the configuration that CANNOT work; that combination
is the trap.

Aggregation over chunks is attention-pooled (a learned query reads the K per-chunk
grids cell by cell) rather than averaged, for the reason MosaicMem gives: averaging
destroys evidence that disagrees, which is exactly the moving content we want left
out rather than smeared in.

With a known pose the code is aggregated in a CANONICAL frame and projected back into
each chunk's own view at decode -- the encode-inverts / decode-reapplies structure.
Without pose the projection is identity and the aggregation happens in view space,
which is only coherent for a static camera.

NON-CAUSAL by design: the whole video is needed before chunk 0 can be decoded. Keep
the causal StaticMemory arms if the online property matters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoStatic(nn.Module):
    def __init__(self, ch: int, grid: int = 4, n_heads: int = 4, pos_dim: int = 32,
                 project: bool = False):
        super().__init__()
        self.ch, self.grid, self.project = int(ch), int(grid), bool(project)

        v = torch.linspace(-1, 1, self.grid)
        vy, vx = torch.meshgrid(v, v, indexing="ij")
        self.register_buffer("cell_xy", torch.stack([vx, vy], -1))       # (g,g,2)
        self.register_buffer("freqs", 2.0 ** torch.arange(pos_dim // 4).float())
        self.pos_proj = nn.Linear(pos_dim, ch)

        self.query = nn.Parameter(torch.randn(1, 1, ch) * 0.02)
        self.k_norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.out = nn.Linear(ch, ch)
        nn.init.zeros_(self.out.weight)     # starts as the plain mean over chunks
        nn.init.zeros_(self.out.bias)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ts(pose):
        return pose[:, :2], torch.exp(pose[:, 2]).unsqueeze(-1)

    def _pos(self, xy):
        a = xy.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        return self.pos_proj(torch.cat([a.sin(), a.cos()], -1).flatten(2))

    def _warp(self, feat, pose_src, pose_dst):
        """Resample a grid from one view into another (2D similarity)."""
        if pose_src is None or pose_dst is None:
            return feat
        B = feat.shape[0]
        q = self.cell_xy.reshape(1, -1, 2).expand(B, -1, -1)
        t_d, s_d = self._ts(pose_dst)
        t_s, s_s = self._ts(pose_src)
        world = s_d.unsqueeze(1) * q + t_d.unsqueeze(1)
        src = (world - t_s.unsqueeze(1)) / s_s.unsqueeze(1).clamp_min(1e-4)
        g = self.grid
        return F.grid_sample(feat, src.reshape(B, g, g, 2), align_corners=True,
                             padding_mode="border")

    # -------------------------------------------------------------- aggregation
    def aggregate(self, per_chunk, poses=None):
        """per_chunk : (B,K,C,g,g) each chunk's static evidence.
        poses : (B,K,3) chunk poses, or None. -> canonical grid (B,C,g,g)."""
        B, K, C, g, _ = per_chunk.shape
        if self.project and poses is not None:
            # bring every chunk into the canonical frame (chunk 0's) before pooling
            canon = poses[:, 0]
            per_chunk = torch.stack(
                [self._warp(per_chunk[:, k], poses[:, k], canon) for k in range(K)], 1)
        base = per_chunk.mean(1)                                    # zero-init anchor
        # attention-pool over the K chunks, per cell
        toks = per_chunk.permute(0, 3, 4, 1, 2).reshape(B * g * g, K, C)
        q = self.query.expand(B * g * g, 1, C)
        att, _ = self.attn(q, self.k_norm(toks), self.k_norm(toks), need_weights=False)
        att = self.out(att).reshape(B, g, g, C).permute(0, 3, 1, 2)
        return base + att

    def to_view(self, canonical, pose_k, pose_canon):
        """Project the canonical code into chunk k's view."""
        if not self.project or pose_k is None:
            return canonical
        return self._warp(canonical, pose_canon, pose_k)


__all__ = ["VideoStatic"]
