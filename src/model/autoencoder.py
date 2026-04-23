import sys
from pathlib import Path

import torch
import torch.nn as nn

_WORKSPACE_ACTAIM = Path(__file__).resolve().parents[3] / "ActAIM3"
if _WORKSPACE_ACTAIM.is_dir():
    actaim_root = str(_WORKSPACE_ACTAIM)
    if actaim_root not in sys.path:
        sys.path.append(actaim_root)

try:
    # Using the exact import path from your RLBench processing script
    from actaim.models.wan.vae.vae2_2 import Wan2_2_VAE
except ImportError:
    Wan2_2_VAE = None


def _load_frozen_vae(vae_pth, device):
    if Wan2_2_VAE is None:
        raise ImportError(
            "Wan2_2_VAE could not be imported. Please make sure the `actaim` "
            "package is installed in the training environment before running train.py."
        )

    vae = Wan2_2_VAE(vae_pth=vae_pth, device=device)

    model = getattr(vae, "model", None)
    if model is not None:
        model.eval()
        model.requires_grad_(False)

    return vae


def _decode_latent_with_vae(vae, latent):
    with torch.no_grad():
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)

        decoded = vae.decode([sample for sample in latent])
        if decoded is None:
            raise RuntimeError("WAN VAE decode returned None.")

        x_recon = torch.stack(decoded, dim=0)
        if x_recon.dim() == 5 and x_recon.shape[2] == 1:
            x_recon = x_recon.squeeze(2)

    return x_recon

class WanFrozenEncoder(nn.Module):
    def __init__(self, vae_pth, device="cuda"):
        super().__init__()
        self.vae = _load_frozen_vae(vae_pth=vae_pth, device=device)

    def forward(self, x):
        """
        x: (B, C, H, W) - A batch of single CLEVRER frames
        Returns: (B, C_out, H_out, W_out) - The dynamics-aware latent
        """
        with torch.no_grad():  # Ensure absolutely no gradients leak into the VAE
            # Video VAEs expect a time dimension: (B, C, T, H, W)
            if x.dim() == 4:
                x = x.unsqueeze(2)

            latents = self.vae.encode([sample for sample in x])
            if latents is None:
                raise RuntimeError("WAN VAE encode returned None.")

            latent = torch.stack(latents, dim=0)
            if latent.dim() == 5 and latent.shape[2] == 1:
                latent = latent.squeeze(2)

        return latent

    def decode(self, latent):
        return _decode_latent_with_vae(self.vae, latent)

class WanFrozenDecoder(nn.Module):
    def __init__(self, vae_pth, device="cuda"):
        super().__init__()
        self.vae = _load_frozen_vae(vae_pth=vae_pth, device=device)

    def forward(self, latent):
        return _decode_latent_with_vae(self.vae, latent)
