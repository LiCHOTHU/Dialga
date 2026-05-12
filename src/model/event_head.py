"""
EventHead — supervised per-slot event detector.

Phase-2 of the v3 event-extractor effort revealed that the per-frame slot
encoder has world-coord RMS ≈ 0.023, which propagates to second-difference
noise (≈ 0.057) larger than typical CLEVRER collision peaks (median 0.029).
SNR < 1 so a hand-coded inertial extractor on encoder q cannot recover events.

Solution (v3 design): supervise a small temporal CNN to predict events from
(encoder q, z_static). Supervision target is the per-frame per-slot event
mask derived from CLEVRER GT collision events. The network learns to handle
the encoder's noise level rather than fighting it.

Input:
    q:        (B, T, K, D)   encoder positions (normalized world units)
    z_static: (B, K, d_s)    pooled identity latent

Output:
    logits:   (B, T, K)      per-frame per-slot event-probability logit
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EventHead(nn.Module):
    """Per-slot 1D-temporal-conv event detector.

    For each slot k we build a (T, D + d_s) per-frame feature stream by
    concatenating the slot's positions over time with its (broadcast)
    z_static identity vector. A small stack of 1D convs along the time axis
    detects local non-smooth motion patterns (kink-like events) while the
    z_static channel lets the head learn slot-specific noise priors
    (different objects move differently).

    Trained with BCEWithLogitsLoss against a dilated `collision_mask`
    label (CLEVRER GT collision events expanded ±`label_dilation` frames).
    """

    def __init__(
        self,
        num_state_dims: int = 2,
        d_static: int = 16,
        hidden: int = 64,
        kernel_size: int = 5,
        depth: int = 2,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.num_state_dims = int(num_state_dims)
        self.d_static = int(d_static)
        in_dim = self.num_state_dims + self.d_static

        layers = []
        ch = in_dim
        for _ in range(int(depth)):
            layers.append(nn.Conv1d(ch, hidden, kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.GELU())
            ch = hidden
        layers.append(nn.Conv1d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, z_static: torch.Tensor) -> torch.Tensor:
        """
        q: (B, T, K, D) ; z_static: (B, K, d_s)
        returns logits: (B, T, K)
        """
        B, T, K, D = q.shape
        z = z_static.unsqueeze(1).expand(B, T, K, -1)              # (B, T, K, d_s)
        x = torch.cat([q, z], dim=-1)                               # (B, T, K, D+d_s)
        x = x.permute(0, 2, 3, 1).reshape(B * K, -1, T)             # (B*K, C, T)
        logit = self.net(x).squeeze(1)                              # (B*K, T)
        return logit.view(B, K, T).transpose(1, 2).contiguous()     # (B, T, K)


def build_event_features(q: torch.Tensor, include_neighbor: bool = False) -> torch.Tensor:
    """(B, W, K, D) → (B, W, K, 3D[+2]): concat [q, v, a] along last dim.

    v[t] = (q[t+1] - q[t-1]) / 2   (centered finite-diff; boundary frames = 0)
    a[t] =  q[t+1] - 2 q[t] + q[t-1]  (2nd diff; boundary frames = 0)

    The unsupervised inertial-baseline extractor only worked because it
    computed the 2nd-difference (Newton's-1st-law residual) explicitly.
    Asking the head to learn finite differencing through Conv1d kernels is
    wasteful — this helper provides it as input so the head's capacity is
    spent on robust thresholding + multi-slot reasoning, not differentiation.

    When include_neighbor=True, appends two pair-aware channels:
      d_nn     = min_{j != k} ||q_k - q_j||  (distance to nearest other slot)
      a_nn     = magnitude of nearest-neighbor's a (impulsive partner signal)
    These let the head learn "I'm in contact with another slot that just
    spiked → fire even if my own a is small" — the structural fix for
    heavy/stationary collision-partner cases.
    """
    q_prev = torch.cat([q[:, :1], q[:, :-1]], dim=1)
    q_next = torch.cat([q[:, 1:], q[:, -1:]], dim=1)
    v = 0.5 * (q_next - q_prev)
    a = q_next - 2 * q + q_prev
    v[:, 0] = 0; v[:, -1] = 0
    a[:, 0] = 0; a[:, -1] = 0
    if not include_neighbor:
        return torch.cat([q, v, a], dim=-1)
    B, W, K, D = q.shape
    diff = q.unsqueeze(3) - q.unsqueeze(2)                    # (B, W, K, K, D)
    dist = diff.norm(dim=-1)                                   # (B, W, K, K)
    eye = torch.eye(K, dtype=torch.bool, device=q.device).view(1, 1, K, K)
    dist = dist.masked_fill(eye, float("inf"))
    d_min, j_min = dist.min(dim=-1)                            # (B, W, K), (B, W, K)
    a_mag = a.norm(dim=-1)                                     # (B, W, K)
    a_nn = a_mag.gather(2, j_min)                              # (B, W, K)
    extra = torch.stack([d_min, a_nn], dim=-1)                  # (B, W, K, 2)
    return torch.cat([q, v, a, extra], dim=-1)


def dilate_label(coll_mask: torch.Tensor, dilation: int = 3) -> torch.Tensor:
    """Temporally dilate a per-(t, slot) bool/float event mask by max-pool.

    coll_mask: (B, T, K)
    dilation:  odd kernel size; 1 = no dilation.
    """
    if dilation <= 1:
        return coll_mask.float()
    if dilation % 2 == 0:
        dilation += 1
    pad = dilation // 2
    cm = coll_mask.float().permute(0, 2, 1)                         # (B, K, T)
    cm = torch.nn.functional.pad(cm, (pad, pad), mode="constant", value=0.0)
    cm = torch.nn.functional.max_pool1d(cm, kernel_size=dilation, stride=1)
    return cm.permute(0, 2, 1).contiguous()


__all__ = ["EventHead", "build_event_features", "dilate_label"]
