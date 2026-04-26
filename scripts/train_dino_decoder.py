"""
Train a lightweight visualization decoder for DinoV2FrozenEncoder.

Input:  DINOv2 patch tokens (B, 384, 16, 16) for dinov2_vits14
Output: RGB frame (B, 3, 224, 224)
Loss:   MSE on pixel reconstruction
Usage:
  python scripts/train_dino_decoder.py \
    --frames_dir /path/to/CLEVRER/train_video \
    --output decoder_ckpt.pth \
    --epochs 20 \
    --lr 1e-3
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class FrameDataset(Dataset):
    def __init__(self, frames_dir, image_size=224, max_samples=0):
        frames_dir = Path(frames_dir)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        paths = sorted(list(frames_dir.rglob("*.png")) + list(frames_dir.rglob("*.jpg")))
        if max_samples > 0:
            paths = paths[:max_samples]
        self.paths = paths
        log.info("FrameDataset: %d frames", len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class DinoVisualizationDecoder(nn.Module):
    """
    ConvTranspose decoder: (B, 384, 16, 16) → (B, 3, 224, 224).
    ~5M parameters. Upsamples 4× via bilinear + conv (14× is handled by final resize).
    """

    def __init__(self, in_channels=384, base_hidden=256, out_size=224):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_hidden, 1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_hidden, base_hidden, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_hidden, base_hidden // 2, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_hidden // 2, base_hidden // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_hidden // 4, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.net(x)
        if out.shape[-1] != self.out_size or out.shape[-2] != self.out_size:
            out = F.interpolate(out, size=(self.out_size, self.out_size),
                                mode="bilinear", align_corners=False)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output", default="decoder_ckpt.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--dinov2_variant", default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model import DinoV2FrozenEncoder

    encoder = DinoV2FrozenEncoder(
        variant=args.dinov2_variant, image_size=args.image_size
    ).to(device)
    encoder.eval()

    decoder = DinoVisualizationDecoder(
        in_channels=encoder.embed_dim,
        out_size=args.image_size,
    ).to(device)

    dataset = FrameDataset(args.frames_dir, image_size=args.image_size,
                           max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0
        for batch_idx, frames in enumerate(loader, 1):
            frames = frames.to(device)
            with torch.no_grad():
                patches = encoder(frames)
            recon = decoder(patches)
            loss = F.mse_loss(recon, frames)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        log.info("Epoch %d/%d | MSE: %.6f | LR: %.2e",
                 epoch, args.epochs, avg_loss, optimizer.param_groups[0]["lr"])
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"decoder": decoder.state_dict(),
                        "variant": args.dinov2_variant,
                        "image_size": args.image_size,
                        "embed_dim": encoder.embed_dim}, args.output)
            log.info("  Saved best decoder to %s", args.output)

    log.info("Done. Best MSE: %.6f", best_loss)


if __name__ == "__main__":
    main()
