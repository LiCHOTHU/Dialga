"""PatchMemory — retrieve-and-compose static memory, after MosaicMem.

WHY THIS REPLACES THE CANVAS. WorldCanvasMemory fuses every observation into one
buffer by a running average (`canvas += w*s; count += w`). That is the explicit-only
design MosaicMem argues against: averaging destroys evidence that disagrees, so
anything that moved is smeared into the scene rather than left out of it. Our own
measurement matched that failure -- the exclusive-attribute probe found z_static
reads *moving* objects better than z_dyn does, i.e. no still/moving separation at all.

MosaicMem instead keeps patches SEPARATE, aligns them into the queried view, and
concatenates them as conditioning tokens so the generator's attention decides, per
region, whether to trust memory or synthesise. Persistence becomes an emergent
property of what attention reads, not something a write gate has to be told.

Its two alignment techniques are complementary (their Table 1: RotErr 0.51 / FID
65.67 with both, vs 0.66 / 75.46 latent-only and 0.70 / 71.89 rope-only), and both
are purely about *where a feature belongs in the query view* -- nothing about them
needs depth or 3D. Transposed to our grid:

  latent  resample the stored grid into the query view (bilinear, fractional
          coords) and tag the tokens with the query lattice coordinates.
          "Move the feature."
  rope    keep the stored cells as they are, but give each one a positional
          embedding of its REPROJECTED coordinate in the query view.
          "Move where the model thinks the feature is."
  both    put both token sets in the bank and let attention pick.

Composition is cross-attention: the current chunk's own cells are the queries, the
aligned memory tokens are keys/values, and a zero-init gated residual writes the
result back -- so at initialisation the module is exactly "no memory" and can only
earn its way in.

Geometry here is the 2D similarity used by synthetic_pan: a cell u in view j maps to
world s_j*u + t_j, and into view q as (s_j*u + t_j - t_q)/s_q. Swapping in a
projective/3D map changes only `_reproject`; the retrieve-and-compose structure is
unchanged, which is the point -- the mechanism is not about dimensionality.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ALIGNS = ("none", "latent", "rope", "both")


class PatchMemory(nn.Module):
    def __init__(self, ch: int, grid: int = 4, n_heads: int = 4,
                 align: str = "both", max_chunks: int = 8, pos_dim: int = 32):
        super().__init__()
        if align not in ALIGNS:
            raise ValueError(f"align must be one of {ALIGNS}, got {align!r}")
        self.ch, self.grid, self.align = int(ch), int(grid), align
        self.max_chunks = int(max_chunks)

        v = torch.linspace(-1, 1, self.grid)
        vy, vx = torch.meshgrid(v, v, indexing="ij")
        self.register_buffer("cell_xy", torch.stack([vx, vy], -1))     # (g,g,2)

        # sinusoidal 2D coordinate features -> additive positional embedding
        self.register_buffer("freqs", 2.0 ** torch.arange(pos_dim // 4).float())
        self.pos_proj = nn.Linear(pos_dim, ch)

        self.vquery = nn.Parameter(torch.randn(1, grid * grid, ch) * 0.02)
        self.q_norm, self.k_norm = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.gate = nn.Linear(2 * ch, ch)
        self.out = nn.Linear(ch, ch)
        nn.init.zeros_(self.out.weight)      # identity at init: memory must earn it
        nn.init.zeros_(self.out.bias)
        nn.init.constant_(self.gate.bias, 0.0)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ts(pose: torch.Tensor):
        return pose[:, :2], torch.exp(pose[:, 2]).unsqueeze(-1)        # (B,2),(B,1)

    def _reproject(self, xy: torch.Tensor, pose_src, pose_q) -> torch.Tensor:
        """Cell coords in the source view -> their coords in the query view.
        xy : (B,N,2). 2D similarity; swap this out for a projective map to go 3D.
        With no known pose (SSv2) this is the identity and alignment falls to
        attention alone -- the pose-free variant."""
        if pose_src is None or pose_q is None:
            return xy
        t_s, s_s = self._ts(pose_src)
        t_q, s_q = self._ts(pose_q)
        world = s_s.unsqueeze(1) * xy + t_s.unsqueeze(1)
        return (world - t_q.unsqueeze(1)) / s_q.unsqueeze(1).clamp_min(1e-4)

    def _pos(self, xy: torch.Tensor) -> torch.Tensor:
        """(B,N,2) coords -> (B,N,ch) additive positional embedding."""
        a = xy.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)             # (B,N,2,F)
        e = torch.cat([a.sin(), a.cos()], -1).flatten(2)                # (B,N,4F)
        return self.pos_proj(e)

    def _resample(self, feat: torch.Tensor, pose_src, pose_q) -> torch.Tensor:
        """Warp a stored grid into the query view (the 'latent' alignment)."""
        B = feat.shape[0]
        if pose_src is None or pose_q is None:
            return feat
        q = self.cell_xy.reshape(1, -1, 2).expand(B, -1, -1)            # query cells
        src = self._reproject(q, pose_q, pose_src)                      # where to read
        g = self.grid
        return F.grid_sample(feat, src.reshape(B, g, g, 2),
                             align_corners=True, padding_mode="border")

    # -------------------------------------------------------------------- build
    def tokens(self, bank, pose_q):
        """bank : list of (feat (B,C,g,g), pose (B,3)). -> (keys (B,N,C), None)."""
        B, C, g, _ = bank[0][0].shape
        base = self.cell_xy.reshape(1, -1, 2).expand(B, -1, -1)         # (B,g*g,2)
        feats, poss = [], []
        for feat, pose_s in bank[-self.max_chunks:]:
            if self.align in ("latent", "both"):
                w = self._resample(feat, pose_s, pose_q)
                feats.append(w.flatten(2).transpose(1, 2))              # (B,g*g,C)
                poss.append(base)                                       # already aligned
            if self.align in ("rope", "both"):
                feats.append(feat.flatten(2).transpose(1, 2))           # raw features
                poss.append(self._reproject(base, pose_s, pose_q))      # warped coords
            if self.align == "none":
                feats.append(feat.flatten(2).transpose(1, 2))
                poss.append(base)
        f = torch.cat(feats, 1)
        p = torch.cat(poss, 1)
        return self.k_norm(f) + self._pos(p)

    # ------------------------------------------------ compose (video-level read)
    def read_video(self, bank, pose_q) -> torch.Tensor:
        """ONE grid for the whole clip: read the entire bank with a learned query
        set instead of anchoring on any single chunk's evidence.

        The per-chunk forward() below returns `cur + gate*attn`, i.e. a code that
        still varies chunk to chunk -- so it costs K*d_static per clip, not
        d_static. Measured: that put PatchMemory at 6656 floats/clip against a
        video-level code's 5504, and the two mechanisms then won about equally at
        their own rate points. This read gives retrieve-and-compose the rate
        saving as well: one code per clip, paid once."""
        B = bank[0][0].shape[0]
        base = self.cell_xy.reshape(1, -1, 2).expand(B, -1, -1)
        q = self.q_norm(self.vquery.expand(B, self.grid ** 2, self.ch)) + self._pos(base)
        kv = self.tokens(bank, pose_q)
        att, _ = self.attn(q, kv, kv, need_weights=False)
        anchor = torch.stack([f for f, _ in bank], 1).mean(1)      # plain mean anchor
        a_t = anchor.flatten(2).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat([a_t, att], -1)))
        out = a_t + gate * self.out(att)
        return out.transpose(1, 2).reshape(B, self.ch, self.grid, self.grid)

    # ------------------------------------------------------------------ compose
    def forward(self, bank, cur: torch.Tensor, pose_q) -> torch.Tensor:
        """cur : (B,C,g,g) this chunk's own collapsed evidence (the query).
        Returns the composed static grid (B,C,g,g)."""
        if not bank:
            return cur
        B, C, g, _ = cur.shape
        base = self.cell_xy.reshape(1, -1, 2).expand(B, -1, -1)
        q = self.q_norm(cur.flatten(2).transpose(1, 2)) + self._pos(base)
        kv = self.tokens(bank, pose_q)
        att, _ = self.attn(q, kv, kv, need_weights=False)
        cur_t = cur.flatten(2).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat([cur_t, att], -1)))
        outn = cur_t + gate * self.out(att)
        return outn.transpose(1, 2).reshape(B, C, g, g)


__all__ = ["PatchMemory", "ALIGNS"]
