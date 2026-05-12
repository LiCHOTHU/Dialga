"""
Direct-acceleration dynamics model.

A drop-in replacement for the autograd-DEL Lagrangian path. Predicts a per-slot
acceleration a = f(q, v, attrs) and integrates with Verlet:

    q_next = 2 * q_curr - q_prev + a * dt^2     (dt = 1 in step units)

Pair contributions are made action-reaction symmetric so total momentum is
conserved up to the visibility mask:

    f_pair(q_i - q_j, v_i, v_j, attr_i, attr_j) = a_i, with a_j = -a_i
    enforced by antisymmetrizing 0.5 * (f(fwd) - f(rev))

This sidesteps the V-degeneracy of the Lagrangian formulation: the network is
trained as a supervised regression on noisy inputs (GNS/MGN style) so it must
predict a coherent vector field over a neighborhood of the GT manifold, rather
than only zeroing a residual at GT points.

Mass head is kept identical to ObjectLagrangian's so CollisionImpulse can
share the interface.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AccelNet(nn.Module):
    def __init__(self, num_state_dims=2, attr_dim=13, hidden=128,
                 use_pair=True, min_mass=1e-3, init_mass=1.0):
        super().__init__()
        self.D = int(num_state_dims)
        self.use_pair = bool(use_pair)
        self.min_mass = float(min_mass)

        self.mass_head = nn.Sequential(
            nn.Linear(attr_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            inv_softplus_init = math.log(math.expm1(max(init_mass - min_mass, 1e-6)))
            nn.init.zeros_(self.mass_head[-1].weight)
            self.mass_head[-1].bias.fill_(inv_softplus_init)

        # Self-acceleration: a_self_i = f_self(q_i, v_i, attrs_i)
        self.f_self = nn.Sequential(
            nn.Linear(2 * self.D + attr_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, self.D),
        )
        nn.init.zeros_(self.f_self[-1].weight)
        nn.init.zeros_(self.f_self[-1].bias)

        if self.use_pair:
            # Pair contribution to slot i: f_pair(Δq, v_i, v_j, attr_i, attr_j).
            # Antisymmetrize to enforce Newton's 3rd law.
            in_dim = self.D + 2 * self.D + 2 * attr_dim
            self.f_pair = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, self.D),
            )
            nn.init.zeros_(self.f_pair[-1].weight)
            nn.init.zeros_(self.f_pair[-1].bias)

    def _mass(self, attrs):
        raw = self.mass_head(attrs).squeeze(-1)
        return F.softplus(raw) + self.min_mass

    def acceleration(self, q, v, attrs, alpha):
        """
        q, v   : (B, K, D)
        attrs  : (B, K, A)
        alpha  : (B, K)   per-slot validity gate (1 = present, 0 = absent)

        Returns a: (B, K, D), zeroed for slots with alpha = 0.
        """
        B, K, D = q.shape
        x_self = torch.cat([q, v, attrs], dim=-1)                    # (B, K, 2D+A)
        a = self.f_self(x_self) * alpha.unsqueeze(-1)                # (B, K, D)

        if self.use_pair and K >= 2:
            idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=q.device)
            q_i, q_j = q[:, idx_i], q[:, idx_j]
            v_i, v_j = v[:, idx_i], v[:, idx_j]
            ai_attr, aj_attr = attrs[:, idx_i], attrs[:, idx_j]
            alpha_ij = (alpha[:, idx_i] * alpha[:, idx_j]).unsqueeze(-1)  # (B, P, 1)
            x_fwd = torch.cat([q_i - q_j, v_i, v_j, ai_attr, aj_attr], dim=-1)
            x_rev = torch.cat([q_j - q_i, v_j, v_i, aj_attr, ai_attr], dim=-1)
            f_ij = 0.5 * (self.f_pair(x_fwd) - self.f_pair(x_rev)) * alpha_ij  # (B, P, D)

            a_pair = q.new_zeros(B, K, D)
            # index_add_ scatters across slot dim
            a_pair.index_add_(1, idx_i, f_ij)
            a_pair.index_add_(1, idx_j, -f_ij)
            a = a + a_pair

        return a

    def forward(self, q, v, attrs, alpha):
        return self.acceleration(q, v, attrs, alpha)


def verlet_step(accel_net, q_prev, q_curr, attrs, alpha_prev, alpha_curr,
                accel_clip=None):
    """
    One Verlet integration step. dt = 1 in step units.

      v_curr = q_curr - q_prev
      a      = accel_net(q_curr, v_curr, attrs, alpha_pair)
      q_next = 2 * q_curr - q_prev + a

    alpha_pair = alpha_prev * alpha_curr (slot must be present in both frames
    for v to be defined). accel_clip optionally caps |a| per-slot to keep
    rollouts finite if the network produces a runaway acceleration.

    Returns q_next: (B, K, D).
    """
    v_curr = q_curr - q_prev
    alpha_pair = alpha_prev * alpha_curr
    a = accel_net.acceleration(q_curr, v_curr, attrs, alpha_pair)
    if accel_clip is not None:
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        norm = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (float(accel_clip) / norm).clamp_max(1.0)
        a = a * scale
    return 2 * q_curr - q_prev + a


__all__ = ["AccelNet", "verlet_step"]
