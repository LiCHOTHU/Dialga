"""ForwardDynamics — Velocity-Verlet integrator on a low-dim state subspace.

Wrapper around a small MLP-acceleration network and a Verlet step, applied
on `state = W_proj(z_dyn)` of dimension `d_state = 2 * d_state_half`. The
state is interpreted as (position, velocity) halves of equal dim; the
accel net produces an acceleration in the position-half subspace.

The W_proj / W_unproj bottleneck is the load-bearing architectural piece:
information in z_dyn that lives in the null space of W_proj is *not*
propagated by the dynamics step, and the unproj step projects back into
a strict d_state-dim subspace of d_dyn, so null-space content is
destroyed each step. Identity stashed in z_dyn therefore drifts toward
zero across rollout, which the predicted-half loss punishes — pushing
identity into z_static instead.

CollisionImpulse from the v21 plan is dropped: with no slot dimension
there are no pairs to act on. Collision events are now the job of
EventHead's additive residual (`g_event`), added to z_dyn *after* the
dynamics step at frames where the gate fires.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardDynamics(nn.Module):
    def __init__(
        self,
        d_dyn: int = 16,
        d_state: int = 8,
        accel_hidden: int = 32,
        dt: float = 1.0,
        no_proj: bool = False,
    ):
        super().__init__()
        if no_proj and d_state != d_dyn:
            raise ValueError(
                f"no_proj requires d_state == d_dyn, got d_state={d_state}, d_dyn={d_dyn}"
            )
        if d_state % 2 != 0:
            raise ValueError(f"d_state must be even (pos|vel split), got {d_state}")
        self.d_dyn = int(d_dyn)
        self.d_state = int(d_state)
        self.d_half = self.d_state // 2
        self.dt = float(dt)
        self.no_proj = bool(no_proj)

        if self.no_proj:
            # Exp 3 ablation: drop the W_proj/W_unproj bottleneck. With d_state==d_dyn
            # the whole z_dyn flows through the Verlet step; any identity content in
            # z_dyn is preserved across the chunk-to-chunk rollout instead of being
            # destroyed by the projection's null space.
            self.W_proj = nn.Identity()
            self.W_unproj = nn.Identity()
        else:
            self.W_proj = nn.Linear(d_dyn, d_state, bias=False)
            self.W_unproj = nn.Linear(d_state, d_dyn, bias=False)

        self.accel_net = nn.Sequential(
            nn.Linear(d_state, accel_hidden),
            nn.SiLU(),
            nn.Linear(accel_hidden, self.d_half),
        )
        # Zero-init the last accel layer so a=0 initially → pure kinematic
        # constant-velocity integration. Gives a stable rollout at step 0 and
        # lets the MLP learn corrections rather than starting from random.
        nn.init.zeros_(self.accel_net[-1].weight)
        nn.init.zeros_(self.accel_net[-1].bias)

    def step(self, z_dyn_t: torch.Tensor) -> torch.Tensor:
        """One Velocity-Verlet step.

        z_dyn_t : (B, d_dyn)  current latent dynamics state
        Returns:
            z_dyn_next : (B, d_dyn)
        """
        if z_dyn_t.dim() != 2 or z_dyn_t.shape[-1] != self.d_dyn:
            raise ValueError(
                f"expected (B, d_dyn={self.d_dyn}), got {tuple(z_dyn_t.shape)}"
            )
        s = self.W_proj(z_dyn_t)                       # (B, d_state)
        pos = s[:, : self.d_half]
        vel = s[:, self.d_half :]
        a = self.accel_net(s)                          # (B, d_half)

        pos_next = pos + vel * self.dt + 0.5 * a * self.dt * self.dt
        vel_next = vel + a * self.dt
        s_next = torch.cat([pos_next, vel_next], dim=-1)  # (B, d_state)
        z_dyn_next = self.W_unproj(s_next)                # (B, d_dyn)
        return z_dyn_next

    def rollout(self, z_dyn_init: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Roll forward n_steps starting from z_dyn_init (one Verlet step per
        iteration; accel re-computed each step). Kept for diagnostics/probes;
        the trainer uses chunk_step for chunk-to-chunk semantics.

        z_dyn_init : (B, d_dyn)
        Returns:
            traj : (B, n_steps, d_dyn) — does NOT include the initial state
        """
        out = []
        z = z_dyn_init
        for _ in range(n_steps):
            z = self.step(z)
            out.append(z)
        return torch.stack(out, dim=1)

    def chunk_step(self, z_dyn_exit: torch.Tensor, T: int) -> torch.Tensor:
        """One CHUNK-TO-CHUNK step: project the chunk's exit state, compute
        ONE acceleration, then analytically unroll the Verlet equations to
        produce T per-frame states for the next chunk.

        z_dyn_exit : (B, d_dyn) — last-frame z_dyn of the observed chunk
        T          : number of latent frames in the predicted chunk
        Returns:
            z_dyn_next : (B, T, d_dyn) — per-frame z_dyn of the predicted chunk
        """
        if z_dyn_exit.dim() != 2 or z_dyn_exit.shape[-1] != self.d_dyn:
            raise ValueError(
                f"expected (B, d_dyn={self.d_dyn}), got {tuple(z_dyn_exit.shape)}"
            )
        s = self.W_proj(z_dyn_exit)                                # (B, d_state)
        pos = s[:, : self.d_half]                                  # (B, d_half)
        vel = s[:, self.d_half :]                                  # (B, d_half)
        a = self.accel_net(s)                                      # (B, d_half) — single accel
        ts = torch.arange(1, T + 1, device=s.device, dtype=s.dtype) * self.dt   # (T,)
        ts_b = ts.view(1, T, 1)                                    # broadcast (1, T, 1)
        pos_t = pos.unsqueeze(1) + vel.unsqueeze(1) * ts_b \
                + 0.5 * a.unsqueeze(1) * ts_b * ts_b               # (B, T, d_half)
        vel_t = vel.unsqueeze(1) + a.unsqueeze(1) * ts_b           # (B, T, d_half)
        s_t = torch.cat([pos_t, vel_t], dim=-1)                    # (B, T, d_state)
        z_dyn_next = self.W_unproj(s_t)                            # (B, T, d_dyn)
        return z_dyn_next


__all__ = ["ForwardDynamics"]
