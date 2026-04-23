import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    """Applies adaLN modulation to a normalized token sequence."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _math_sdp_kernel_context(x):
    if not x.is_cuda:
        return nullcontext()

    # DEL training differentiates through attention gradients, which requires
    # second-order support that fused SDPA kernels do not provide in this env.
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    )


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # Keep the branches close to adaLN-Zero, but not exactly asleep. Exact
        # zero gates make early higher-order DEL gradients nearly uninformative.
        hidden = hidden_size
        nn.init.constant_(self.adaLN_modulation[1].bias[2 * hidden : 3 * hidden], 0.01)
        nn.init.constant_(self.adaLN_modulation[1].bias[5 * hidden : 6 * hidden], 0.01)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        with _math_sdp_kernel_context(normed_x):
            attn_out, _ = self.attn(normed_x, normed_x, normed_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        normed_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


def _inverse_softplus(value):
    """Returns x such that softplus(x) ~= value for positive value."""
    return math.log(math.expm1(float(value)))


class StructuredLagrangianHead(nn.Module):
    """
    Predicts L = T - V with less latent-space degeneracy.

    V is a sum of per-patch scalar potentials, giving local force gradients.
    T uses one positive scalar mass per sample instead of a per-latent-coordinate
    diagonal, preventing the kinetic term from becoming an arbitrary rescaler.
    """

    def __init__(self, hidden_size, min_mass=1e-3, init_mass=1.0):
        super().__init__()
        self.min_mass = float(min_mass)

        self.potential_head = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 1),
        )
        self.mass_head = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 1),
        )

        nn.init.normal_(self.potential_head[1].weight, std=0.02)
        nn.init.zeros_(self.potential_head[1].bias)
        nn.init.normal_(self.mass_head[1].weight, std=0.02)
        nn.init.constant_(
            self.mass_head[1].bias,
            _inverse_softplus(max(init_mass - self.min_mass, 1e-6)),
        )

    def forward(self, cls_out, patch_tokens):
        patch_potential = self.potential_head(patch_tokens).squeeze(-1)
        potential = patch_potential.sum(dim=-1)
        mass = F.softplus(self.mass_head(cls_out)).squeeze(-1) + self.min_mass
        return mass, potential, patch_potential


class DiTLagrangian(nn.Module):
    """
    Structured discrete Lagrangian over latent states.

    L_d(q_{t-1}, q_t) = T(q_t - q_{t-1}; m(q_mid)) - V(q_mid)
    where q_mid = 0.5 * (q_{t-1} + q_t).
    """

    def __init__(
        self,
        latent_channels=16,
        latent_h=16,
        latent_w=16,
        patch_size=2,
        hidden_size=256,
        depth=4,
        num_heads=8,
        action_dim=0,
    ):
        super().__init__()

        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError(
                "latent_h and latent_w must be divisible by patch_size for patch embedding."
            )

        self.latent_channels = int(latent_channels)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.patch_size = int(patch_size)
        self.action_dim = int(action_dim)
        self.flat_latent_dim = self.latent_channels * self.latent_h * self.latent_w

        self.patch_embed = nn.Conv2d(
            self.latent_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.patch_grid_h = self.latent_h // self.patch_size
        self.patch_grid_w = self.latent_w // self.patch_size
        self.num_patches = self.patch_grid_h * self.patch_grid_w

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))

        if self.action_dim > 0:
            self.condition_embedder = nn.Sequential(
                nn.Linear(self.action_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.default_condition = nn.Parameter(torch.zeros(1, hidden_size))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads) for _ in range(depth)]
        )
        self.output_head = StructuredLagrangianHead(hidden_size=hidden_size)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        if hasattr(self, "default_condition"):
            nn.init.normal_(self.default_condition, std=0.02)

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled):
        self.gradient_checkpointing = bool(enabled)

    def _validate_latents(self, q_prev, q_curr):
        if q_prev.shape != q_curr.shape:
            raise ValueError(
                "q_prev and q_curr must have the same shape, got "
                f"{q_prev.shape} and {q_curr.shape}."
            )

        expected_latent_shape = (self.latent_channels, self.latent_h, self.latent_w)
        if tuple(q_prev.shape[1:]) != expected_latent_shape:
            raise ValueError(
                "Latent shape does not match the model configuration. "
                f"Expected {expected_latent_shape}, got {tuple(q_prev.shape[1:])}."
            )

    def _condition(self, batch_size, action):
        if self.action_dim > 0:
            if action is None:
                raise ValueError("An action tensor must be provided when action_dim > 0.")
            return self.condition_embedder(action)
        return self.default_condition.expand(batch_size, -1)

    def _encode_midpoint(self, q_mid, action):
        batch_size = q_mid.shape[0]
        x = self.patch_embed(q_mid)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        c = self._condition(batch_size, action)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and (x.requires_grad or c.requires_grad)
            ):
                x = checkpoint(
                    lambda x_in, c_in, block=block: block(x_in, c_in),
                    x,
                    c,
                    use_reentrant=False,
                )
            else:
                x = block(x, c)

        return x

    def compute_components(self, q_prev, q_curr, action=None):
        """
        Returns differentiable Lagrangian components for diagnostics/training.
        """
        self._validate_latents(q_prev, q_curr)
        batch_size = q_prev.shape[0]

        v = q_curr - q_prev
        q_mid = 0.5 * (q_prev + q_curr)
        x = self._encode_midpoint(q_mid, action)

        cls_out = x[:, 0]
        patch_tokens = x[:, 1:]
        mass, potential, patch_potential = self.output_head(cls_out, patch_tokens)

        v_flat = v.reshape(batch_size, -1)
        kinetic = 0.5 * mass * v_flat.pow(2).sum(dim=-1)
        lagrangian = kinetic - potential

        return {
            "lagrangian": lagrangian,
            "kinetic": kinetic,
            "potential": potential,
            "mechanical_energy": kinetic + potential,
            "mass": mass,
            "patch_potential": patch_potential,
        }

    def forward(self, q_prev, q_curr, action=None):
        return self.compute_components(q_prev, q_curr, action=action)["lagrangian"]


__all__ = ["DiTLagrangian"]
