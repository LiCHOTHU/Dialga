import torch
from torch import nn


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer.

    In the current repo it is used only during the state-learning stage to keep
    the learned latent from becoming low-rank or degenerate while the encoder and
    dynamics are optimized together.
    """

    def __init__(self, knots=17, num_proj=256):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be at least 2.")
        if num_proj <= 0:
            raise ValueError("num_proj must be > 0.")

        self.num_proj = int(num_proj)
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / float(knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        if proj.dim() != 3:
            raise ValueError(f"Expected projections with shape (T, B, D), got {tuple(proj.shape)}.")

        projection_matrix = torch.randn(
            proj.shape[-1],
            self.num_proj,
            device=proj.device,
            dtype=proj.dtype,
        )
        projection_matrix = projection_matrix / projection_matrix.norm(p=2, dim=0).clamp_min(1e-8)

        x_t = (proj @ projection_matrix).unsqueeze(-1) * self.t.to(proj.dtype)
        err = (
            (x_t.cos().mean(dim=-3) - self.phi.to(proj.dtype)).square()
            + x_t.sin().mean(dim=-3).square()
        )
        statistic = (err @ self.weights.to(proj.dtype)) * proj.shape[-2]
        return statistic.mean()


__all__ = ["SIGReg"]
