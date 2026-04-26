"""
Linear-probe diagnostic: how much physical state is linearly recoverable
from a candidate encoder?

Usage:
  python scripts/probe_encoder.py \
    --frames_dir /path/to/CLEVRER/train_video \
    --annotation_dir /path/to/CLEVRER/annotations \
    --wan_vae_ckpt /path/to/Wan2.2_VAE.pth \
    --n_train 5000 \
    --n_val 1000 \
    --output probe_results.csv

For each encoder, encodes all frames → flattens to (N, D) → fits sklearn Ridge
for each (object_slot, coord_dim) → reports R² on the val split.
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: aligned (frame, annotation) pairs
# ---------------------------------------------------------------------------

class FrameStatePairDataset(torch.utils.data.Dataset):
    """
    Returns (frame_tensor, positions_flat, velocities_flat) for individual
    time steps, aligned from CLEVRER annotations + video frame folders.

    positions_flat / velocities_flat: (max_objects * 3,) world-xyz coordinates.
    Frames returned in [-1, 1] range (torchvision ToTensor + Normalize).
    """

    def __init__(self, annotation_dir, frames_dir, max_objects=8, image_size=128, max_samples=0):
        import json
        import re
        from torchvision import transforms

        self.max_objects = max_objects
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        annotation_dir = Path(annotation_dir)
        frames_dir = Path(frames_dir)

        # Discover all annotation json files
        ann_files = sorted(list(annotation_dir.rglob("*.json")))
        if not ann_files:
            raise FileNotFoundError(f"No annotation jsons found under {annotation_dir}")

        samples = []
        for ann_path in ann_files:
            with ann_path.open() as f:
                ann = json.load(f)
            scene_idx = ann["scene_index"]
            vid_name = ann["video_filename"]

            # Find matching frame folder
            match = re.search(r"video_(\d+)\.mp4$", vid_name)
            if match is None:
                continue
            vid_idx = int(match.group(1))
            subdir_lo = (vid_idx // 1000) * 1000
            subdir = f"video_{subdir_lo:05d}-{subdir_lo + 1000:05d}"
            frame_folder = frames_dir / subdir / f"video_{vid_idx:05d}"
            if not frame_folder.exists():
                # try flat layout
                frame_folder = frames_dir / f"video_{vid_idx:05d}"
            if not frame_folder.exists():
                continue

            frame_files = sorted(frame_folder.glob("*.png")) + sorted(frame_folder.glob("*.jpg"))
            if not frame_files:
                continue

            traj = ann["motion_trajectory"]
            obj_props = ann["object_property"]
            obj_id_to_slot = {int(p["object_id"]): i for i, p in enumerate(
                sorted(obj_props, key=lambda x: x["object_id"])
            )}

            for frame_idx, frame_info in enumerate(traj):
                if frame_idx >= len(frame_files):
                    break
                pos3d = np.zeros((max_objects, 3), dtype=np.float32)
                for obj in frame_info["objects"]:
                    slot = obj_id_to_slot.get(int(obj["object_id"]))
                    if slot is not None and slot < max_objects:
                        loc = obj["location"]
                        pos3d[slot] = [loc[0], loc[1], loc[2]]
                samples.append((frame_files[frame_idx], pos3d))

        if max_samples > 0:
            samples = samples[:max_samples]
        self.samples = samples
        log.info("FrameStatePairDataset: %d samples", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        frame_path, pos3d = self.samples[idx]
        img = Image.open(frame_path).convert("RGB")
        frame = self.transform(img)
        pos_flat = torch.tensor(pos3d.flatten(), dtype=torch.float32)
        return frame, pos_flat


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------

def build_encoders(wan_vae_ckpt, device):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.model import WanFrozenEncoder, DinoV2FrozenEncoder, FrozenDINOAutoencoder

    encoders = {}

    if wan_vae_ckpt and Path(wan_vae_ckpt).exists():
        encoders["wan_vae_frozen"] = WanFrozenEncoder(vae_pth=wan_vae_ckpt, device=device)
    else:
        log.warning("WAN VAE checkpoint not found or not provided; skipping wan_vae_frozen")

    encoders["dinov2_vits14"] = DinoV2FrozenEncoder(variant="dinov2_vits14", image_size=224).to(device)
    encoders["dinov2_vitb14"] = DinoV2FrozenEncoder(variant="dinov2_vitb14", image_size=224).to(device)

    encoders["dino_projected_vits14"] = FrozenDINOAutoencoder(
        model_name="dinov2_vits14", input_size=126, latent_channels=64
    ).to(device)

    return encoders


@torch.no_grad()
def encode_dataset(encoder, loader, device):
    all_feats, all_pos = [], []
    for frames, pos_flat in loader:
        frames = frames.to(device)
        feats = encoder(frames)
        all_feats.append(feats.flatten(1).cpu().numpy())
        all_pos.append(pos_flat.numpy())
    return np.concatenate(all_feats, axis=0), np.concatenate(all_pos, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def probe_encoder(name, X_train, Y_train, X_val, Y_val):
    log.info("  Fitting Ridge probe on %d train / %d val samples, D=%d ...",
             X_train.shape[0], X_val.shape[0], X_train.shape[1])
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_val)
    r2 = r2_score(Y_val, Y_pred, multioutput="uniform_average")
    mse = float(np.mean((Y_pred - Y_val) ** 2))
    return r2, mse


def main():
    parser = argparse.ArgumentParser(description="Linear probe encoder diagnostic")
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--wan_vae_ckpt", default="")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", default="probe_results.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    total_samples = args.n_train + args.n_val
    dataset = FrameStatePairDataset(
        annotation_dir=args.annotation_dir,
        frames_dir=args.frames_dir,
        max_samples=total_samples,
    )
    if len(dataset) < total_samples:
        log.warning("Only %d samples available (requested %d)", len(dataset), total_samples)

    n_train = min(args.n_train, len(dataset) - 1)
    n_val = min(args.n_val, len(dataset) - n_train)
    train_ds = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_ds = torch.utils.data.Subset(dataset, list(range(n_train, n_train + n_val)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    encoders = build_encoders(args.wan_vae_ckpt, device)

    rows = []
    header = ["encoder", "pos_R2", "pos_MSE", "feature_dim"]

    print(f"\n{'encoder':<35} {'pos_R2':>8} {'pos_MSE':>10} {'feat_dim':>10}")
    print("-" * 70)

    for enc_name, encoder in encoders.items():
        encoder.eval()
        log.info("Encoding with %s ...", enc_name)
        try:
            X_train, Y_train = encode_dataset(encoder, train_loader, device)
            X_val, Y_val = encode_dataset(encoder, val_loader, device)
        except Exception as e:
            log.error("  Failed to encode with %s: %s", enc_name, e)
            continue

        feat_dim = X_train.shape[1]
        log.info("  Feature dim: %d", feat_dim)

        r2, mse = probe_encoder(enc_name, X_train, Y_train, X_val, Y_val)
        rows.append([enc_name, f"{r2:.4f}", f"{mse:.6f}", str(feat_dim)])
        print(f"{enc_name:<35} {r2:>8.4f} {mse:>10.6f} {feat_dim:>10}")

    print("-" * 70)

    out_path = Path(args.output)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
