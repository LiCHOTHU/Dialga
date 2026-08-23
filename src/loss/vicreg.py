"""VICReg variance + covariance regularizers (Bardes et al. 2022).

The internals inspection (2026-06-12) found the latent is starved: effective
dimensionality (participation ratio) is ~15/96 for z_static and ~10.6/96 for
z_dyn, so only ~110 of 964 allocated floats carry information. Nothing in the
loss rewarded spreading information across dimensions, so the optimizer settled
into a low-rank solution and the extra dims added by scaling stayed empty.

Two terms, applied to the batch matrix of a latent z (N rows, D dims):
  * variance hinge  - relu(gamma - std_d) averaged over dims; pushes every dim
                      to have std >= gamma so none collapses to a constant.
  * covariance      - sum of squared off-diagonal covariances / D; decorrelates
                      dimensions so each carries independent information.

Together they drive the participation ratio up toward D. We deliberately drop
VICReg's third (invariance) term: that role is already played by InfoNCE on
z_static and the recon/pred objectives on z_dyn.
"""

from __future__ import annotations

import torch


def vicreg_var_cov(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    """
    z : (N, D) batch of latent vectors (flatten any time axis into N first).
    Returns (var_loss, cov_loss), both scalars.
    """
    if z.dim() != 2:
        raise ValueError(f"vicreg expects (N, D), got {tuple(z.shape)}")
    N, D = z.shape
    if N < 2:
        zero = z.sum() * 0.0
        return zero, zero
    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + eps)                      # (D,)
    var_loss = torch.relu(gamma - std).mean()
    # Correlation (not raw covariance) off-diagonal: scale-invariant, bounded in
    # [-1, 1], so the penalty cannot explode as the latent magnitude grows. (Raw
    # covariance grows with variance^2 and blew up to ~8 in testing.)
    cov = (zc.T @ zc) / (N - 1)                                # (D, D)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    off_diag = corr - torch.diag(torch.diagonal(corr))
    cov_loss = off_diag.pow(2).sum() / (D * (D - 1))
    return var_loss, cov_loss


def cross_decorr(za: torch.Tensor, zd: torch.Tensor, eps: float = 1e-4):
    """Cross-code independence penalty: mean squared cross-CORRELATION between the
    static code `za` (N, Ds) and the (time-pooled) dynamics code `zd` (N, Dd).

    Standardizes each code per-dim, forms the Ds x Dd cross-correlation matrix, and
    returns the mean squared entry. Minimizing it pushes z_dyn to be linearly
    uninformative about z_static -> identity (which lives in z_static via InfoNCE)
    is driven OUT of z_dyn. Label-free; scale-invariant (bounded in [0,1]).
    """
    if za.dim() != 2 or zd.dim() != 2:
        raise ValueError(f"cross_decorr expects (N,Ds),(N,Dd); got {tuple(za.shape)},{tuple(zd.shape)}")
    N = za.shape[0]
    if N < 2:
        return za.sum() * 0.0
    za = (za - za.mean(0, keepdim=True)) / (za.std(0, keepdim=True) + eps)
    zd = (zd - zd.mean(0, keepdim=True)) / (zd.std(0, keepdim=True) + eps)
    xcorr = (za.T @ zd) / (N - 1)                                 # (Ds, Dd)
    return xcorr.pow(2).mean()


__all__ = ["vicreg_var_cov", "cross_decorr"]
