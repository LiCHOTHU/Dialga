"""WorldCanvasMemory — an EXPLICIT world-frame buffer for the static scene.

The other memory modes keep the static code in *view* coordinates: one g x g grid,
the size of what the camera sees right now. Under a moving camera that is the wrong
container -- the scene is bigger than any single view, so a view-sized buffer can
only ever hold the current window and must overwrite whatever left the frame.

This module keeps the scene where it belongs: a G x G canvas in WORLD coordinates,
covering `extent` times the view's field, with a companion coverage counter.

    write   every frame of every chunk is warped OUT to world coordinates using the
            known camera pose and accumulated into the canvas (+ its coverage count)
    read    the current chunk warps the canvas BACK into its own view

so chunk k can read scene structure that only chunk 0 ever observed. This is the
MUSt3R / Spann3R accumulation property, made concrete at feature resolution, and it
is the explicit half of the explicit/implicit pair (compose with `gru` for a hybrid:
explicit geometric registration, implicit learned gating on the readout).

Both directions are a single `grid_sample`. Writing is done as a *backward* sample
onto the canvas lattice -- for each canvas cell we ask which view cell it came from,
which is differentiable and avoids a scatter:

    view p  <->  world  s*p + t          (matches synthetic_pan's affine convention:
    canvas c <-> world  extent * c        the warped view at p shows the original at
                                          s*p + t, so the view sees the world region
                                          centred at t with half-extent s)

Pose is (t_x, t_y, log_scale) in normalised grid units, relativised to the VIDEO's
first frame so every chunk writes into one common frame.

RESOLUTION MATTERS, and quietly. A write and a read are two bilinear resamplings, so
the canvas must OVERSAMPLE the view or the round trip destroys the code. Measured
write-then-read-at-the-same-pose error, which should be ~0:

    view 4x4 -> canvas  8x8   err 0.665   coverage 0.250  (wrong: quantised)
    view 4x4 -> canvas 16x16  err 0.082   coverage 0.391  (= (2/3.2)^2, exact)
    view 4x4 -> canvas 32x32  err 0.069   coverage 0.391
    view 8x8 -> canvas 16x16  err 0.043   coverage 0.391

The view spans 2 world units and the canvas spans 2*extent, so `canvas_mult` must be
at least ~2*extent for the canvas to be no coarser than the view, and ~2x that again
to survive the double resampling. Hence the default of 4 rather than 2. At a 4x4
static grid this is genuinely coarse geometry -- feature-level registration, not
metric 3D, and it should never be described as the latter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldCanvasMemory(nn.Module):
    def __init__(self, ch: int, grid: int = 4, extent: float = 1.6,
                 canvas_mult: int = 4, learned_gate: bool = False):
        super().__init__()
        self.ch = int(ch)
        self.grid = int(grid)
        self.extent = float(extent)
        self.G = int(grid * canvas_mult)

        c = torch.linspace(-1, 1, self.G)
        cy, cx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("canvas_xy", torch.stack([cx, cy], -1))   # (G,G,2)
        v = torch.linspace(-1, 1, self.grid)
        vy, vx = torch.meshgrid(v, v, indexing="ij")
        self.register_buffer("view_xy", torch.stack([vx, vy], -1))     # (g,g,2)

        # optional learned write gate: lets the model down-weight evidence it does
        # not trust (a mover) instead of averaging it into the scene.
        self.gate = nn.Conv2d(self.ch, 1, 1) if learned_gate else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 2.0)      # sigmoid(2) ~ 0.88, write by default

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ts(pose: torch.Tensor):
        """pose (B,3) = (t_x, t_y, log_scale) -> (t (B,1,1,2), s (B,1,1,1))."""
        t = pose[:, :2].view(-1, 1, 1, 2)
        s = torch.exp(pose[:, 2]).view(-1, 1, 1, 1)
        return t, s

    def empty(self, B: int, device, dtype):
        canvas = torch.zeros(B, self.ch, self.G, self.G, device=device, dtype=dtype)
        count = torch.zeros(B, 1, self.G, self.G, device=device, dtype=dtype)
        return canvas, count

    # -------------------------------------------------------------------- write
    def write(self, state, feat: torch.Tensor, pose: torch.Tensor):
        """feat : (B,T,C,g,g) this chunk's per-frame static features.
        pose : (B,T,3) per-frame camera pose, video-relative."""
        B, T = feat.shape[:2]
        canvas, count = state
        for t in range(T):
            tr, sc = self._ts(pose[:, t])
            # for each canvas cell: which view cell did it come from?
            world = self.extent * self.canvas_xy.unsqueeze(0)          # (1,G,G,2)
            p = (world - tr) / sc.clamp_min(1e-4)                      # (B,G,G,2)
            inside = ((p.abs() <= 1.0).all(-1, keepdim=True)
                      .permute(0, 3, 1, 2).to(feat.dtype))             # (B,1,G,G)
            samp = F.grid_sample(feat[:, t], p, align_corners=True,
                                 padding_mode="zeros")                 # (B,C,G,G)
            w = inside
            if self.gate is not None:
                w = w * torch.sigmoid(self.gate(samp))
            canvas = canvas + samp * w
            count = count + w
        return canvas, count

    # --------------------------------------------------------------------- read
    def read(self, state, pose: torch.Tensor) -> torch.Tensor:
        """Warp the world canvas back into this chunk's view. pose : (B,3)."""
        canvas, count = state
        world_mem = canvas / count.clamp_min(1e-4)
        tr, sc = self._ts(pose)
        c = (sc * self.view_xy.unsqueeze(0) + tr) / self.extent        # (B,g,g,2)
        return F.grid_sample(world_mem, c, align_corners=True,
                             padding_mode="border")                    # (B,C,g,g)

    def coverage(self, state) -> torch.Tensor:
        """Fraction of the canvas ever observed — how much scene the memory holds."""
        _, count = state
        return (count > 0).float().mean(dim=(1, 2, 3))


__all__ = ["WorldCanvasMemory"]
