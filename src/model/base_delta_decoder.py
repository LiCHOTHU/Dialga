"""BaseDeltaDecoder — a decoder whose wiring *forces* the static/dynamic split.

Why this exists (measured, not assumed). With the standard SpatialGridDecoder --
concatenate [upsampled z_static grid, per-frame z_dyn grid, coords] and let a conv
stack mix them -- the static code turns out to be nearly decorative:

    decode chunk 0 with ...          val latent MSE
    full                             0.0391
    z_static ZEROED                  0.0430   (+10%)
    z_dyn    ZEROED                  0.0560   (+43%)
    z_static from a DIFFERENT video  0.0464   (+19%)

z_dyn is a full-resolution per-frame 8x8 grid, so it can simply re-encode every
frame in its entirety; nothing makes the model put persistent content in z_static.
Projecting z_dyn onto the zero-temporal-mean subspace does NOT fix it (+4%): a
nonlinear decoder recovers time-constant structure from a zero-mean code (its
magnitude pattern is constant even when its sign is not).

So the constraint has to hold in OUTPUT space:

    x_hat_t = Base(z_static) + [ Delta(z_dyn_t) - mean_t Delta(z_dyn_t) ]

The delta branch never sees z_static, and its temporal mean is subtracted after
the nonlinearity, so it is EXACTLY zero-mean over the chunk. Therefore

    mean_t x_hat = Base(z_static)

identically: the temporally-constant part of the reconstruction is produced by
z_static alone, and z_dyn can only describe deviation from it. That is the
"static = what does not move, dynamic = what changes" split compiled into the
architecture instead of hoped for from a loss term. On this data the constant part
carries ~54% of the chunk's latent energy, so it is a real job, not a token one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack(in_ch: int, hidden: int, out_ch: int, depth: int, n_groups: int):
    layers = [nn.Conv2d(in_ch, hidden, 3, padding=1),
              nn.GroupNorm(n_groups, hidden), nn.SiLU()]
    for _ in range(max(0, depth - 1)):
        layers += [nn.Conv2d(hidden, hidden, 3, padding=1),
                   nn.GroupNorm(n_groups, hidden), nn.SiLU()]
    head = nn.Conv2d(hidden, out_ch, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class BaseDeltaDecoder(nn.Module):
    def __init__(self, latent_ch: int = 48, d_static: int = 96,
                 static_grid: int = 4, d_dyn: int = 256, dyn_grid: int = 8,
                 hidden_ch: int = 384, spatial_size: int = 8,
                 n_groups: int = 8, depth: int = 3):
        super().__init__()
        g2, g2d = static_grid ** 2, dyn_grid ** 2
        if d_static % g2 or d_dyn % g2d:
            raise ValueError("d_static/d_dyn must divide their grid areas")
        self.c_static, self.c_dyn = d_static // g2, d_dyn // g2d
        self.static_grid, self.dyn_grid = static_grid, dyn_grid
        self.d_dyn, self.latent_ch = d_dyn, latent_ch
        self.spatial_size = spatial_size

        s = spatial_size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, s), torch.linspace(-1, 1, s),
                                indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys], 0))     # (2,H,W)

        self.base = _stack(self.c_static + 2, hidden_ch, latent_ch, depth, n_groups)
        self.delta = _stack(self.c_dyn + 2, hidden_ch, latent_ch, depth, n_groups)

    def forward(self, z_static_grid: torch.Tensor, z_dyn: torch.Tensor,
                pose_emb=None) -> torch.Tensor:
        B, _, gh, gw = z_static_grid.shape
        T = z_dyn.shape[1]
        s = self.spatial_size

        c1 = self.coords.unsqueeze(0).expand(B, -1, -1, -1)
        up_s = F.interpolate(z_static_grid, size=(s, s), mode="bilinear",
                             align_corners=False)
        base = self.base(torch.cat([up_s, c1], 1))                   # (B,C,H,W)

        zdg = z_dyn.reshape(B * T, self.c_dyn, self.dyn_grid, self.dyn_grid)
        up_d = F.interpolate(zdg, size=(s, s), mode="bilinear", align_corners=False)
        cT = self.coords.unsqueeze(0).expand(B * T, -1, -1, -1)
        delta = self.delta(torch.cat([up_d, cT], 1)).reshape(B, T, self.latent_ch, s, s)
        delta = delta - delta.mean(dim=1, keepdim=True)   # exactly zero-mean over T

        out = base.unsqueeze(1) + delta                              # (B,T,C,H,W)
        return out.permute(0, 2, 1, 3, 4).contiguous()               # (B,C,T,H,W)


__all__ = ["BaseDeltaDecoder"]
