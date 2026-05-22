"""InfoNCE — symmetric contrastive loss with in-batch negatives.

Used in v5.1 as L_static_consist replacement: pull z_static of two chunks
from the same video together, push z_static of different videos apart.
Replaces the trivially-collapsible MSE consistency term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE between two views z_a, z_b ∈ (B, D).

    Position i in z_a and position i in z_b are treated as a positive pair;
    all other positions in the batch are negatives.

    Returns scalar loss; bounded above by ~log(2B) for random inputs and
    approaches 0 when z_a, z_b are perfectly aligned (and other rows are
    pushed apart).
    """
    if z_a.shape != z_b.shape or z_a.dim() != 2:
        raise ValueError(
            f"z_a and z_b must both be (B, D), got {tuple(z_a.shape)} and {tuple(z_b.shape)}"
        )
    B = z_a.shape[0]
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits_ab = z_a @ z_b.T / temperature    # (B, B)
    logits_ba = z_b @ z_a.T / temperature
    targets = torch.arange(B, device=z_a.device)
    loss_ab = F.cross_entropy(logits_ab, targets)
    loss_ba = F.cross_entropy(logits_ba, targets)
    return 0.5 * (loss_ab + loss_ba)


__all__ = ["info_nce"]
