import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2FrozenEncoder(nn.Module):
    """
    Frozen DINOv2 patch encoder with no trainable components.

    Outputs raw patch tokens reshaped to (B, C, H_patch, W_patch).
    For dinov2_vits14 at 224x224: (B, 384, 16, 16).

    Expects frames in [-1, 1] (standard CLEVRER dataset normalization).
    Converts internally to ImageNet normalization before passing to DINOv2.
    Has no decoder — use a separately-trained visualization decoder.
    """

    def __init__(self, variant="dinov2_vits14", image_size=224):
        super().__init__()
        self.image_size = int(image_size)
        self.model = torch.hub.load("facebookresearch/dinov2", variant, verbose=False)
        self.model.eval()
        self.model.requires_grad_(False)
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.grid = self.image_size // self.patch_size

        self.can_decode = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    @torch.no_grad()
    def forward(self, x):
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        x = (x + 1.0) / 2.0
        x = (x - self.imagenet_mean) / self.imagenet_std
        feats = self.model.forward_features(x)
        patches = feats["x_norm_patchtokens"]
        b, n, c = patches.shape
        return patches.transpose(1, 2).reshape(b, c, self.grid, self.grid)

    def decode(self, latent, enable_grad=False):
        raise NotImplementedError(
            "DinoV2FrozenEncoder has no native decoder. "
            "Train a separate visualization decoder (scripts/train_dino_decoder.py)."
        )


class FrozenDINOAutoencoder(nn.Module):
    """
    Frozen DINOv2 encoder with a small trainable latent projection and decoder.

    The encoder is frozen. The projection/decoder are trainable so the latent can
    be adapted to the current dynamics model and visualized in pixel space.
    """

    def __init__(self, model_name="dinov2_vits14", input_size=126, latent_channels=64):
        super().__init__()
        self.input_size = int(input_size)
        self.patch_grid = self.input_size // 14
        self.feature_dim = 384
        self.latent_channels = int(latent_channels)

        self.encoder = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.projector = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.latent_channels, kernel_size=1),
        )

        hidden = max(self.latent_channels, 64)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 2, hidden // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden // 4, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected image batch with shape (B, C, H, W), got {tuple(x.shape)}.")

        x_resized = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            features = self.encoder.forward_features(x_resized)
            patch_tokens = features["x_norm_patchtokens"]

        batch_size = x.shape[0]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            batch_size,
            self.feature_dim,
            self.patch_grid,
            self.patch_grid,
        )
        return self.projector(patch_tokens)

    def decode(self, latent, enable_grad=False):
        grad_context = torch.enable_grad if enable_grad else torch.no_grad
        with grad_context():
            recon = self.decoder(latent)
            recon = F.interpolate(recon, size=(128, 128), mode="bilinear", align_corners=False)
        return recon


__all__ = ["DinoV2FrozenEncoder", "FrozenDINOAutoencoder"]
