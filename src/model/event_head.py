"""Event-related heads for v5.1 (chunk-wise).

  EventHead       z_dyn (B, d_dyn)            -> z_event (B, d_event)
  GEvent          z_event (B, d_event)        -> residual (B, d_dyn)
  GatePredictor   z_dyn_obs (B, d_dyn)        -> logit (B,)   scalar per chunk

One event-residual per chunk transition. Gate is "did a collision happen
in the next chunk." No temporal axis — Conv1d-over-time is gone with the
per-frame z_dyn it relied on.

Zero-init rule: only GEvent.proj is zero-init. EventHead.net[-1] is NOT
zero-init (composition would otherwise have dead gradient on both paths).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EventHead(nn.Module):
    """Predict d_event-dim event residual from chunk z_dyn."""

    def __init__(self, d_dyn: int = 16, d_event: int = 4, hidden: int = 32):
        super().__init__()
        self.d_dyn = int(d_dyn)
        self.d_event = int(d_event)
        self.net = nn.Sequential(
            nn.Linear(d_dyn, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_event),
        )

    def forward(self, z_dyn: torch.Tensor) -> torch.Tensor:
        if z_dyn.dim() != 2 or z_dyn.shape[-1] != self.d_dyn:
            raise ValueError(f"expected (B, d_dyn={self.d_dyn}), got {tuple(z_dyn.shape)}")
        return self.net(z_dyn)                  # (B, d_event)


class GEvent(nn.Module):
    """Tiny linear lift d_event -> d_dyn. Zero-init keeps the additive
    correction off at step 0."""

    def __init__(self, d_event: int = 4, d_dyn: int = 16):
        super().__init__()
        self.proj = nn.Linear(d_event, d_dyn, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, z_event: torch.Tensor) -> torch.Tensor:
        return self.proj(z_event)  # (B, d_dyn)


class GatePredictor(nn.Module):
    """Predict a single scalar gate logit per chunk."""

    def __init__(self, d_dyn: int = 16, hidden: int = 32):
        super().__init__()
        self.d_dyn = int(d_dyn)
        self.net = nn.Sequential(
            nn.Linear(d_dyn, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_dyn: torch.Tensor) -> torch.Tensor:
        if z_dyn.dim() != 2 or z_dyn.shape[-1] != self.d_dyn:
            raise ValueError(f"expected (B, d_dyn={self.d_dyn}), got {tuple(z_dyn.shape)}")
        return self.net(z_dyn).squeeze(-1)      # (B,)


__all__ = ["EventHead", "GEvent", "GatePredictor"]
