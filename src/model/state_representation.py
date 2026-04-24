import torch
from torch import nn


class ResidualStateProjector(nn.Module):
    """
    Lightweight trainable adapter on top of frozen VAE latents.

    It is initialized as an exact identity so the baseline behavior is preserved
    at step 0, while still giving SIGReg trainable parameters to shape.
    """

    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )

        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected latent tensor with shape (B, C, H, W), got {tuple(x.shape)}.")
        return x + self.residual(x)


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer.

    Input shape: (T, B, D), where T is time, B is batch, and D is the flattened
    latent dimensionality.
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
        projection_matrix = projection_matrix / projection_matrix.norm(
            p=2,
            dim=0,
        ).clamp_min(1e-8)

        x_t = (proj @ projection_matrix).unsqueeze(-1) * self.t.to(proj.dtype)
        err = (
            (x_t.cos().mean(dim=-3) - self.phi.to(proj.dtype)).square()
            + x_t.sin().mean(dim=-3).square()
        )
        statistic = (err @ self.weights.to(proj.dtype)) * proj.shape[-2]
        return statistic.mean()


__all__ = ["ResidualStateProjector", "SIGReg"]
