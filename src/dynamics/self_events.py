"""Self-supervised event teacher.

For the general (no-GT-collision) regime, event labels are generated on-the-fly
from the inertial-baseline residual. The principle (from the v3 roadmap):
events are where the *inertial* prediction fails, not where a *trained* dynamics
model fails (the latter signal vanishes at convergence).

Given predicted positions q, the Newton-1st-law residual is

    a_i^t = q_i^{t+1} - 2 q_i^t + q_i^{t-1}     (2nd finite difference)

Its magnitude ||a|| spikes at impulsive contacts. Z-scoring it per slot over
the window and passing through a sigmoid yields a soft event label that the
EventHead can be trained against — without any GT collision annotation.

This is domain-portable: works on any system with smooth nominal motion. The
CLEVRER GT collision_mask is replaced by this teacher.
"""

from __future__ import annotations

import torch


def event_soft_from_residual(
    q: torch.Tensor,
    alpha: torch.Tensor,
    z_thresh: float = 1.5,
    sharpness: float = 2.5,
    label_dilation: int = 1,
) -> torch.Tensor:
    """Produce soft per-(t, slot) event labels from positional 2nd-diff.

    q     : (B, T, K, D) predicted slot positions (encoder output)
    alpha : (B, T, K) visibility in {0, 1} (or soft mass-of-mask)

    Returns:
        event_soft : (B, T, K) in [0, 1], detached (teacher — no gradient flow)

    Boundary frames (t=0 and t=T-1) get label 0 — the residual is undefined
    there. Frames where any of (t-1, t, t+1) are invisible also get 0.
    """
    B, T, K, D = q.shape
    if T < 3:
        return q.new_zeros(B, T, K)

    # 2nd finite difference. Detach so the teacher doesn't backprop into q
    # through its own label — the head learns to fit a frozen target derived
    # from the current encoder.
    q_d = q.detach()
    a = q_d[:, 2:] - 2 * q_d[:, 1:-1] + q_d[:, :-2]   # (B, T-2, K, D)
    r = a.norm(dim=-1)                                # (B, T-2, K)

    # Per-slot z-score, gated by visibility of all three frames in the triple.
    gate = alpha[:, :-2] * alpha[:, 1:-1] * alpha[:, 2:]   # (B, T-2, K)
    cnt = gate.sum(dim=1, keepdim=True).clamp_min(1.0)     # (B, 1, K)
    masked_r = r * gate
    mu = masked_r.sum(dim=1, keepdim=True) / cnt
    var = ((r - mu) ** 2 * gate).sum(dim=1, keepdim=True) / cnt
    sigma = var.sqrt().clamp_min(1e-4)
    z = (r - mu) / sigma                                   # (B, T-2, K)

    event_inner = torch.sigmoid((z - z_thresh) * sharpness) * gate

    # Pad back to T frames.
    event = q.new_zeros(B, T, K)
    event[:, 1:-1] = event_inner

    # Optional dilation: a collision actually spans a few frames; dilate the
    # teacher label by max-pooling over a ±d window so the supervised target
    # gives the head some slack.
    if label_dilation > 0:
        k = 2 * int(label_dilation) + 1
        x = event.transpose(1, 2).reshape(B * K, 1, T)     # (BK, 1, T)
        x = torch.nn.functional.max_pool1d(x, kernel_size=k, stride=1, padding=label_dilation)
        event = x.view(B, K, T).transpose(1, 2)

    return event.detach()


__all__ = ["event_soft_from_residual"]
