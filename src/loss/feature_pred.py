"""masked_feature_loss — cosine distance on masked cells only (MAETok L_mask).

MAETok computes its auxiliary reconstruction loss only at masked positions:
    L_mask = sum_j || M * (y_hat_j - y_j) ||^2
and uses cosine distance for the DINOv2 feature target. We follow the cosine
form: 1 - cos(pred, target) averaged over masked cells.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_feature_loss(pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """
    pred   : (B, D, T, H, W)  aux-decoder output (channel-first, decoder layout)
    target : (B, T, H, W, D)  cached teacher features (channel-last, cache layout)
    mask   : (B, T, H, W)     bool, True at masked cells
    Returns scalar: mean over masked cells of (1 - cosine similarity).
    """
    pred_cl = pred.permute(0, 2, 3, 4, 1)                       # (B, T, H, W, D)
    if pred_cl.shape != target.shape:
        raise ValueError(f"pred {tuple(pred_cl.shape)} != target {tuple(target.shape)}")
    if not mask.any():
        return pred.sum() * 0.0
    p = pred_cl[mask]                                           # (M, D)
    t = target[mask]
    cos = F.cosine_similarity(p.float(), t.float(), dim=-1)
    return (1.0 - cos).mean()


__all__ = ["masked_feature_loss"]
