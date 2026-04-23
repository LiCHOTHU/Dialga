import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def modulate(x, shift, scale):
    """Applies the adaLN modulation to the normalized tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # adaLN modulation layer outputs 6 parameters: shift, scale, gate for both sub-layers
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # adaLN-Zero Initialization
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        # Wake the residual branches slightly so the Lagrangian has meaningful
        # higher-order structure from the start.
        hidden = hidden_size
        nn.init.constant_(self.adaLN_modulation[1].bias[2 * hidden : 3 * hidden], 0.01)
        nn.init.constant_(self.adaLN_modulation[1].bias[5 * hidden : 6 * hidden], 0.01)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention Block
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP Block
        normed_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(normed_x)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

def _inverse_softplus(value):
    """Returns x such that softplus(x) ~= value for positive value."""
    return math.log(math.expm1(value))


class StructuredLagrangianHead(nn.Module):
    """
    Predicts the components of a structured Lagrangian L = T - V.

    The kinetic term uses a positive diagonal mass vector while the
    potential term is a scalar field over the midpoint state.
    """
    def __init__(self, hidden_size, flat_latent_dim, min_mass=1e-3, init_mass=1.0):
        super().__init__()
        self.min_mass = min_mass

        self.potential_head = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, 1),
        )
        self.mass_head = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, flat_latent_dim),
        )

        nn.init.normal_(self.potential_head[1].weight, std=0.02)
        nn.init.constant_(self.potential_head[1].bias, 0.01)
        nn.init.normal_(self.mass_head[1].weight, std=0.02)
        nn.init.constant_(
            self.mass_head[1].bias,
            _inverse_softplus(max(init_mass - min_mass, 1e-6)),
        )

    def forward(self, cls_out):
        potential = self.potential_head(cls_out).squeeze(-1)
        mass_diag = F.softplus(self.mass_head(cls_out)) + self.min_mass
        return mass_diag, potential

class DiTLagrangian(nn.Module):
    """
    A structured discrete Lagrangian that constrains the DiT to learn
    midpoint potential energy and a positive diagonal mass matrix.
    """
    def __init__(self, 
                 latent_channels=16, 
                 latent_h=16, 
                 latent_w=16, 
                 patch_size=2, 
                 hidden_size=256, 
                 depth=4, 
                 num_heads=8,
                 action_dim=0):  # Set to 0 for CLEVRER, >0 for RLBench
        super().__init__()

        if latent_h % patch_size != 0 or latent_w % patch_size != 0:
            raise ValueError(
                "latent_h and latent_w must be divisible by patch_size for patch embedding."
            )

        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.action_dim = action_dim
        self.flat_latent_dim = latent_channels * latent_h * latent_w
        
        # 1. Patch Embedder over the midpoint state q_mid
        self.patch_embed = nn.Conv2d(
            latent_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (latent_h // patch_size) * (latent_w // patch_size)
        
        # 2. Positional Encoding & CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        
        # 3. Action Condition Embedder
        if action_dim > 0:
            self.condition_embedder = nn.Sequential(
                nn.Linear(action_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        else:
            # For pure physics (CLEVRER), we learn a static default condition vector
            self.default_condition = nn.Parameter(torch.zeros(1, hidden_size))
        
        # 4. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 5. Structured output head for T - V
        self.output_head = StructuredLagrangianHead(
            hidden_size=hidden_size,
            flat_latent_dim=self.flat_latent_dim,
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        if hasattr(self, "default_condition"):
            nn.init.normal_(self.default_condition, std=0.02)
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled):
        self.gradient_checkpointing = bool(enabled)

    def forward(self, q_prev, q_curr, action=None):
        """
        Calculates a structured discrete Lagrangian
        L_d(q_{t-1}, q_t, a_t) = T(v; M(q_mid)) - V(q_mid),
        where q_mid = 0.5 * (q_{t-1} + q_t) and v = q_t - q_{t-1}.
        """
        B = q_prev.shape[0]

        if q_prev.shape != q_curr.shape:
            raise ValueError(
                f"q_prev and q_curr must have the same shape, got {q_prev.shape} and {q_curr.shape}."
            )

        expected_latent_shape = (self.latent_channels, self.latent_h, self.latent_w)
        if tuple(q_prev.shape[1:]) != expected_latent_shape:
            raise ValueError(
                "Latent shape does not match the model configuration. "
                f"Expected {expected_latent_shape}, got {tuple(q_prev.shape[1:])}."
            )
        
        # --- Structured mechanics features ---
        v = q_curr - q_prev
        q_mid = 0.5 * (q_prev + q_curr)

        # --- Prepare Spatial Features ---
        x = self.patch_embed(q_mid)                      # (B, hidden_size, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)                 # (B, num_patches, hidden_size)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)            # (B, num_patches + 1, hidden_size)
        x = x + self.pos_embed
        
        # --- Prepare Conditioning ---
        if self.action_dim > 0:
            if action is None:
                raise ValueError("An action tensor must be provided when action_dim > 0.")
            c = self.condition_embedder(action)          # (B, hidden_size)
        else:
            c = self.default_condition.expand(B, -1)     # (B, hidden_size)
            
        # --- DiT Backbone ---
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and (x.requires_grad or c.requires_grad):
                x = checkpoint(
                    lambda x_in, c_in, block=block: block(x_in, c_in),
                    x,
                    c,
                    use_reentrant=False,
                )
            else:
                x = block(x, c)
            
        # --- Structured T - V Output ---
        cls_out = x[:, 0]
        mass_diag, potential = self.output_head(cls_out)
        v_flat = v.reshape(B, -1)
        kinetic = 0.5 * torch.sum(mass_diag * v_flat.pow(2), dim=-1)
        energy = kinetic - potential

        return energy
