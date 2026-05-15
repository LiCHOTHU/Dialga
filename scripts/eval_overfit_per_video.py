"""Per-video overfit eval for the slot pipeline (rungs 4 and 5).

Loads stage2.pt and reports, per CLEVRER video:
  - encoder state RMSE in world units (q vs GT)
  - 1-step rollout RMSE (predict t=2 from GT t=0,1)
  - full-window rollout RMSE at t=2..W-1 (autoregressive from GT t=0,1)
  - SlotPixelDecoder reconstruction pixel MSE (encoder-predicted q + z_static)

Dispatches between AccelNet (verlet) and ObjectLagrangian (autograd-DEL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.accel_net import AccelNet, verlet_step
from src.model.slot_lagrangian import (
    CollisionImpulse,
    SlotPixelDecoder,
)
from scripts.train_slot import (
    _autoregressive_rollout_eval,
    build_encoder,
    encode_window,
    pool_z_static,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to stage2.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    pos_norm = float(cfg.dataset.pos_normalize)
    dyn_type = "accel"
    d_static = int(cfg.model.get("d_static", 16))
    print(f"Loaded {args.ckpt}")
    print(f"  encoder={cfg.model.encoder_type} | dynamics={dyn_type} | d_static={d_static}")

    # ---- Dataset (5 videos)
    dataset = ClevrerPairedDataset(
        data_dir=str(cfg.dataset.data_dir),
        annotation_dir=str(cfg.dataset.annotation_dir),
        split=str(cfg.dataset.split),
        window_length=int(cfg.training.window_length),
        frames_per_video=int(cfg.dataset.video_num_frames),
        windows_per_video=int(cfg.training.windows_per_video),
        max_videos=int(cfg.training.max_videos),
        max_objects=int(cfg.dataset.max_objects),
        coordinate_mode=str(cfg.dataset.coordinate_mode),
        image_size=int(cfg.dataset.image_size),
        seed=int(cfg.training.seed),
    )
    attr_dim = dataset.attr_dim

    # ---- Models
    encoder = build_encoder(cfg, attr_dim).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    dyn = AccelNet(
        num_state_dims=int(cfg.model.num_state_dims),
        attr_dim=d_static,
        hidden=int(cfg.model.lagrangian_hidden),
        use_pair=bool(cfg.model.lagrangian_use_pair),
    )
    dyn.load_state_dict(ckpt["lagrangian_state_dict"])
    dyn.to(device).eval()

    impulse = CollisionImpulse(
        attr_dim=d_static,
        hidden=int(cfg.model.get("impulse_hidden", 64)),
    )
    if "impulse_state_dict" in ckpt:
        impulse.load_state_dict(ckpt["impulse_state_dict"])
    impulse.to(device).eval()

    decoder = SlotPixelDecoder(
        num_state_dims=int(cfg.model.num_state_dims),
        attr_dim=d_static,
        max_objects=int(cfg.dataset.max_objects),
        image_size=int(cfg.dataset.image_size),
        hidden=int(cfg.model.decoder_hidden),
        slot_embed=int(cfg.model.decoder_slot_embed),
        num_blocks=int(cfg.model.decoder_blocks),
        grid_size=int(cfg.model.decoder_grid_size),
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.to(device).eval()

    # ---- Pick first window for each of the 5 videos
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= 5:
            break
    if not seen:
        for i in range(len(dataset)):
            s = dataset[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= 5:
                break
    samples = list(seen.values())
    batch = paired_collate(samples)

    frames = batch["frames"].to(device)
    gt_pos = batch["positions"].to(device).float() / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device)
    collisions = batch["collisions"]
    N, W = frames.shape[:2]
    K = gt_pos.shape[2]
    print(f"\n5-video batch: N={N}, W={W}, K={K}, image={frames.shape[-2:]}")
    print(f"GT collisions per video: {[len(c) for c in collisions]}\n")

    # ---- Encode (all videos at once)
    with torch.no_grad():
        q_pred, z_static_pf = encode_window(encoder, frames, attrs)
    z_static = pool_z_static(z_static_pf, visib)  # (N, K, d_static)

    # ---- Per-video metrics
    header = f"{'Video':>6} | {'enc RMSE':>9} | {'1-step RMSE':>11} | {'rollout RMSE @ t=W-1':>20} | {'recon MSE':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for n, s in enumerate(samples):
        vid = s["video_id"]

        # encoder state RMSE (world)
        d_q = (q_pred[n:n+1] - gt_pos[n:n+1]).pow(2).sum(dim=-1)  # (1, W, K)
        m = visib[n:n+1]
        enc_mse = (d_q * m).sum() / m.sum().clamp_min(1.0)
        enc_rmse = (enc_mse.sqrt() * pos_norm).item()

        # one-step rollout: GT t=0,1 → predict t=2
        with torch.no_grad():
            q2 = verlet_step(
                dyn,
                gt_pos[n:n+1, 0], gt_pos[n:n+1, 1],
                z_static[n:n+1],
                visib[n:n+1, 0], visib[n:n+1, 1],
                accel_clip=0.5,
            )
        d_1 = (q2 - gt_pos[n:n+1, 2]).pow(2).sum(dim=-1)
        denom_1 = visib[n:n+1, 2].sum().clamp_min(1.0)
        one_rmse = (((d_1 * visib[n:n+1, 2]).sum() / denom_1).sqrt() * pos_norm).item()

        # full-window autoregressive rollout
        roll = _autoregressive_rollout_eval(
            dyn, impulse,
            gt_pos[n:n+1], visib[n:n+1], z_static[n:n+1],
            collisions_per_sample=[collisions[n]],
            solver_alpha=float(cfg.training.get("solver_alpha", 0.1)),
            solver_steps=max(int(cfg.training.get("solver_steps", 1)), 2),
            direction_clip=0.5,
            dynamics_type=dyn_type,
        )
        d_roll_last = (roll[:, W-1] - gt_pos[n:n+1, W-1]).pow(2).sum(dim=-1)
        denom_last = visib[n:n+1, W-1].sum().clamp_min(1.0)
        roll_rmse = (((d_roll_last * visib[n:n+1, W-1]).sum() / denom_last).sqrt() * pos_norm).item()

        # SlotPixelDecoder reconstruction (encoder predictions + pooled z_static)
        with torch.no_grad():
            flat_q = q_pred[n].reshape(W, *q_pred.shape[2:])
            z_rep = z_static[n].unsqueeze(0).expand(W, *z_static.shape[1:])
            v_rep = visib[n].reshape(W, *visib.shape[2:])
            recon = decoder(flat_q, z_rep, v_rep)
        recon_mse = F.mse_loss(recon, frames[n]).item()

        print(f"{vid:>6} | {enc_rmse:>9.4f} | {one_rmse:>11.4f} | "
              f"{roll_rmse:>20.4f} | {recon_mse:>10.6f}")
        rows.append((vid, enc_rmse, one_rmse, roll_rmse, recon_mse))

    # ---- Aggregate
    n_rows = len(rows)
    print("-" * len(header))
    print(f"{'mean':>6} | "
          f"{sum(r[1] for r in rows)/n_rows:>9.4f} | "
          f"{sum(r[2] for r in rows)/n_rows:>11.4f} | "
          f"{sum(r[3] for r in rows)/n_rows:>20.4f} | "
          f"{sum(r[4] for r in rows)/n_rows:>10.6f}")

    # ---- Per-frame rollout RMSE for video 0 (drift profile)
    print(f"\nDrift profile (video {samples[0]['video_id']}, RMSE world units, all-slot mean):")
    n = 0
    roll = _autoregressive_rollout_eval(
        dyn, impulse,
        gt_pos[n:n+1], visib[n:n+1], z_static[n:n+1],
        collisions_per_sample=[collisions[n]],
        solver_alpha=float(cfg.training.get("solver_alpha", 0.1)),
        solver_steps=max(int(cfg.training.get("solver_steps", 1)), 2),
        direction_clip=0.5,
        dynamics_type=dyn_type,
    )
    for t in range(W):
        d_t = (roll[:, t] - gt_pos[n:n+1, t]).pow(2).sum(dim=-1)
        denom_t = visib[n:n+1, t].sum().clamp_min(1.0)
        rmse_t = (((d_t * visib[n:n+1, t]).sum() / denom_t).sqrt() * pos_norm).item()
        marker = " (GT)" if t < 2 else ""
        print(f"  t={t}: RMSE = {rmse_t:.4f}{marker}")


if __name__ == "__main__":
    main()
