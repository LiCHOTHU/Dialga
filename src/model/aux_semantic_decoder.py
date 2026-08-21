"""AuxSemanticDecoder — throwaway feature decoder for masked semantic prediction.

MAETok's auxiliary decoders share the main decoder's design but are shallow
and predict frozen-teacher features (DINOv2) instead of pixels; they exist
only during training and are discarded afterwards.

Fork C port: structurally identical to LatentDecoder — per-frame Linear lift
from [z_static; z_dyn[t]] to a hidden 8x8 grid, one refining conv, 1x1
projection — but the output channel count is the DINO feature dim (384 for
dinov2-small) rather than the Wan latent's 48. The final projection is
re-initialised to a SMALL random value (std 1e-3) rather than LatentDecoder's
exact zeros: the masked-feature loss is a cosine, whose gradient diverges as
1/|pred| at pred=0, so an exactly-zero head produces a ~1e7 gradient spike on
the first step (which global grad-clipping would then let dominate the whole
update). Small-but-nonzero keeps the aux path effectively inert at epoch 0
with finite, well-scaled gradients.

Forcing the prediction to come from (z_static, z_dyn) — the SAME pooled codes
the main decoder consumes — is the entire point: the codes themselves must
carry the semantic content of the masked cells.
"""

from __future__ import annotations

from src.model.latent_decoder import LatentDecoder


class AuxSemanticDecoder(LatentDecoder):
    def __init__(
        self,
        d_feat: int = 384,
        d_static: int = 32,
        d_dyn: int = 16,
        hidden_ch: int = 128,
        chunk_size_lat: int = 9,
        spatial_size: int = 8,
        n_groups: int = 8,
    ):
        super().__init__(
            latent_ch=d_feat,
            d_static=d_static,
            d_dyn=d_dyn,
            hidden_ch=hidden_ch,
            chunk_size_lat=chunk_size_lat,
            spatial_size=spatial_size,
            n_groups=n_groups,
        )
        self.d_feat = int(d_feat)
        # Overwrite LatentDecoder's zero-init (see module docstring).
        import torch.nn as nn
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.normal_(self.out_proj.bias, std=1e-3)
        # forward(z_static (B,D_s), z_dyn (B,T,D_d)) -> (B, d_feat, T, H, W)


__all__ = ["AuxSemanticDecoder"]
