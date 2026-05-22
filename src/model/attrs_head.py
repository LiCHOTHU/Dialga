"""AttrsHead — weak supervised classifier on z_static for (color, material, shape).

Exp 1 (2026-05-21): identity probes show contrastive InfoNCE recovers color
identity but stalls on material and shape (Δ +0.07 / +0.09, both short of the
+0.15 threshold). We add a light-touch supervised head with weight 0.05 to
provide the missing gradient signal *without* dominating the representation.

The head reads z_static (B, D_s) and predicts three categorical groups.
Loss is sum-of-CE on the per-video MODAL label (most-common class across
visible slots) — same target the probe evaluates on.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttrsHead(nn.Module):
    def __init__(
        self,
        d_static: int,
        n_color: int,
        n_material: int,
        n_shape: int,
        hidden: int = 0,
    ):
        super().__init__()
        if hidden > 0:
            self.trunk = nn.Sequential(nn.Linear(d_static, hidden), nn.SiLU())
            d_in = hidden
        else:
            self.trunk = nn.Identity()
            d_in = d_static
        self.head_color    = nn.Linear(d_in, n_color)
        self.head_material = nn.Linear(d_in, n_material)
        self.head_shape    = nn.Linear(d_in, n_shape)

    def forward(self, z_static: torch.Tensor) -> dict:
        h = self.trunk(z_static)
        return {
            "color":    self.head_color(h),
            "material": self.head_material(h),
            "shape":    self.head_shape(h),
        }


__all__ = ["AttrsHead"]
