import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_state_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint["config"], checkpoint


def build_autoencoder(cfg, device):
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.model import LeWMPatchAutoencoder

    autoencoder = LeWMPatchAutoencoder(
        image_size=int(cfg["model"]["input_image_size"]),
        patch_size=int(cfg["model"]["lewm_patch_size"]),
        embed_dim=int(cfg["model"]["lewm_embed_dim"]),
        latent_channels=int(cfg["model"]["lewm_latent_channels"]),
        depth=int(cfg["model"]["lewm_encoder_depth"]),
        num_heads=int(cfg["model"]["lewm_num_heads"]),
        mlp_ratio=float(cfg["model"]["lewm_mlp_ratio"]),
    ).to(device)
    return autoencoder


def main():
    parser = argparse.ArgumentParser(description="Train pixel decoder on a frozen learned latent state.")
    parser.add_argument("--state_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_videos", type=int, default=128)
    parser.add_argument("--windows_per_video", type=int, default=4)
    parser.add_argument("--window_length", type=int, default=6)
    parser.add_argument("--max_frames", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg, checkpoint = load_state_checkpoint(args.state_ckpt, device)

    import sys
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.data.clevrer_sequence import ClevrerSequenceWindowDataset

    autoencoder = build_autoencoder(cfg, device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"], strict=False)
    autoencoder.encoder.requires_grad_(False)
    autoencoder.projector.requires_grad_(False)
    autoencoder.decoder.requires_grad_(True)
    autoencoder.train()

    dataset = ClevrerSequenceWindowDataset(
        data_dir=cfg["dataset"]["data_dir"],
        window_length=args.window_length,
        max_frames=args.max_frames,
        windows_per_video=args.windows_per_video,
        max_videos=args.max_videos,
        seed=int(cfg["training"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(autoencoder.decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_steps = 0
        for frames in loader:
            frames = frames.to(device)
            with torch.no_grad():
                latents = autoencoder(frames.flatten(0, 1))
            recon = autoencoder.decode(latents)
            target = frames.flatten(0, 1)
            loss = F.mse_loss(recon, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        log.info("Epoch %d/%d | recon MSE: %.6f", epoch, args.epochs, avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "decoder_state_dict": autoencoder.decoder.state_dict(),
                    "source_state_ckpt": args.state_ckpt,
                    "best_loss": best_loss,
                },
                output_dir / "best_decoder.pt",
            )

    log.info("Done. Best decoder MSE: %.6f", best_loss)


if __name__ == "__main__":
    main()
