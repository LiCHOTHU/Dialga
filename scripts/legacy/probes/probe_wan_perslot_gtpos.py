"""Experiment 1 follow-up: per-slot raw-latent probe with GT-position pooling.

Distinguishes "per-slot info isn't in the Wan latent" vs "encoder is the bottleneck."

Pipeline:
    1. Project each (object_xy_world, z≈0.35) → image pixel (u, v) with the
       CLEVRER fixed Blender camera.
    2. Downsample to the 8×8 latent grid (×1/16).
    3. Pool the raw Wan latent (48 ch, T_lat, 8, 8) with a Gaussian soft-mask
       centered at the slot's projected position (σ = 1.5 latent units),
       averaged over visible frames → per-slot feature ∈ R^48.
    4. Linear probe (color/material/shape) using the same 50/50 by-video
       split as the other probes.

Sanity-checks the projection by reporting (visible-fraction-inside-frame)
before running probes; aborts if calibration looks broken.

Run:
    python scripts/probe_wan_perslot_gtpos.py \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W12 \\
        --ckpt outputs/iter22c_attrs_20260519_030044/trajectory.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from scripts.legacy.train_trajectory import CachedLatentDataset, collate


ATTR_GROUPS = [
    ("color",    0,                                       len(COLOR_VOCAB),                                          COLOR_VOCAB),
    ("material", len(COLOR_VOCAB),                        len(COLOR_VOCAB) + len(MATERIAL_VOCAB),                    MATERIAL_VOCAB),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB),  len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB), SHAPE_VOCAB),
]


# -------------------- CLEVRER camera projection --------------------
# CLEVR's render_images.py defaults (CLEVRER inherits): camera at this fixed
# location, looks at origin, square render auto-cropped to 128. Lens 35 mm,
# sensor 32 mm.
CAM_POS    = np.array([7.4811, -6.5072, 5.3437], dtype=np.float64)
CAM_TARGET = np.array([0.0,     0.0,     0.0   ], dtype=np.float64)
WORLD_UP   = np.array([0.0,     0.0,     1.0   ], dtype=np.float64)
SENSOR_MM  = 32.0
FOCAL_MM   = 35.0
OBJ_Z      = 0.35     # typical CLEVRER object-center height (radius of small sphere/cube)
IMG_SIZE   = 128


def _camera_frame():
    fwd = CAM_TARGET - CAM_POS; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, WORLD_UP); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def project_world_xy_to_pixel(xy: np.ndarray) -> np.ndarray:
    """xy : (..., 2) world. Returns (..., 2) pixel coords (u, v) in [0, IMG_SIZE]
    (may be outside range if off-screen). z is fixed to OBJ_Z.
    """
    fwd, right, up = _camera_frame()
    world = np.concatenate([xy, np.full(xy.shape[:-1] + (1,), OBJ_Z)], axis=-1)
    rel = world - CAM_POS
    cam_x = (rel * right).sum(-1)
    cam_y = (rel * up).sum(-1)
    cam_z = (rel * fwd).sum(-1)                  # depth (positive = in front of camera)
    # FOV from sensor & lens. Square image (auto-fit) → fov_v = fov_h.
    fov_h = 2 * math.atan((SENSOR_MM / 2) / FOCAL_MM)
    half_w = np.tan(fov_h / 2) * cam_z
    half_h = half_w
    u_norm = cam_x / half_w                      # [-1, 1] when in frame
    v_norm = -cam_y / half_h                     # flip y for image-down convention
    u = (u_norm + 1) * 0.5 * IMG_SIZE
    v = (v_norm + 1) * 0.5 * IMG_SIZE
    return np.stack([u, v], axis=-1)


# -------------------- per-slot feature extraction --------------------


def gaussian_mask(center_xy_latent: np.ndarray, H: int = 8, sigma: float = 1.5) -> np.ndarray:
    """Returns (..., H, H) soft mask, summing to 1 per slot."""
    grid = np.arange(H, dtype=np.float64) + 0.5    # bin centers
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    # center_xy_latent shape (..., 2)
    cx = center_xy_latent[..., 0:1, None]
    cy = center_xy_latent[..., 1:2, None]
    d2 = (xx[None, ...] - cx) ** 2 + (yy[None, ...] - cy) ** 2
    raw = np.exp(-d2 / (2 * sigma * sigma))
    mass = raw.sum(axis=(-2, -1), keepdims=True).clip(min=1e-8)
    return raw / mass


def pool_latent_per_slot(latent: np.ndarray, positions_w: np.ndarray,
                         visibility: np.ndarray, slot_mask: np.ndarray,
                         sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """latent:     (48, T_lat, 8, 8)
       positions_w:(T_pix, K, 2) world XY
       visibility: (T_pix, K) bool
       slot_mask:  (K,) bool

    Returns:
        feats:        (K, 48) — soft-mask pool over space, mean over T_lat,
                                with the spatial center = mean of visible
                                projected positions for that slot.
        in_frame_frac:(K,)    — fraction of visible frames whose projected
                                position landed in [0, IMG_SIZE]^2 (sanity).
    """
    K = slot_mask.shape[0]
    C, T_lat, H, W = latent.shape
    assert H == W == 8

    # Mean latent over T_lat — temporal info is small here.
    latent_mean = latent.mean(axis=1)              # (48, 8, 8)

    feats = np.zeros((K, C), dtype=np.float32)
    in_frame_frac = np.zeros(K, dtype=np.float32)
    for k in range(K):
        if not bool(slot_mask[k]):
            continue
        vis = visibility[:, k].astype(bool)
        if not vis.any():
            continue
        pos_world = positions_w[vis, k]            # (T_visible, 2)
        pix = project_world_xy_to_pixel(pos_world) # (T_visible, 2) in [0, 128]
        in_frame = ((pix >= 0) & (pix < IMG_SIZE)).all(axis=-1)
        in_frame_frac[k] = in_frame.mean()
        # Use only frames where projection is inside the image, but fall back
        # to all visible frames if all projections are off-frame (rare).
        chosen = pix[in_frame] if in_frame.any() else pix
        mean_pix = chosen.mean(axis=0)
        latent_xy = mean_pix / (IMG_SIZE / H)      # → [0, 8]
        mask = gaussian_mask(latent_xy[None, :], H=H, sigma=sigma)[0]   # (H, H)
        # Apply mask: weighted sum of latent_mean (C, H, W) by mask (H, W) → (C,)
        feats[k] = (latent_mean * mask).sum(axis=(-2, -1))
    return feats, in_frame_frac


# -------------------- driver --------------------


def gather(ds, batch_size, num_workers, sigma):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    vids, feats_all, attrs_all, smasks, in_frame_all = [], [], [], [], []
    for batch in loader:
        B = batch["latent"].shape[0]
        for b in range(B):
            f, infr = pool_latent_per_slot(
                batch["latent"][b].numpy(),
                batch["positions"][b].numpy(),
                batch["visibility"][b].numpy(),
                batch["slot_mask"][b].numpy(),
                sigma=sigma,
            )
            feats_all.append(f)
            in_frame_all.append(infr)
        vids.append(batch["video_id"].clone())
        attrs_all.append(batch["attrs"].clone())
        smasks.append(batch["slot_mask"].clone().bool())
    return {
        "video_id":     torch.cat(vids),
        "feats":        torch.from_numpy(np.stack(feats_all)),
        "attrs":        torch.cat(attrs_all),
        "slot_mask":    torch.cat(smasks),
        "in_frame":     torch.from_numpy(np.stack(in_frame_all)),
    }


def per_slot_pairs(feat, attrs, slot_mask, video_ids, video_set, lo, hi):
    keep_win = torch.tensor([int(v.item()) in video_set for v in video_ids], dtype=torch.bool)
    feat = feat[keep_win]; attrs_ = attrs[keep_win]; mask = slot_mask[keep_win]
    xs, ys = [], []
    N, K = mask.shape
    for n in range(N):
        for k in range(K):
            if not bool(mask[n, k].item()):
                continue
            oh = attrs_[n, k, lo:hi]
            if oh.sum().item() < 0.5:
                continue
            xs.append(feat[n, k])
            ys.append(int(oh.argmax().item()))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_linear(x_tr, y_tr, x_te, y_te, nc, epochs=2000, lr=1e-2, wd=1e-4, device="cuda"):
    probe = nn.Linear(x_tr.shape[1], nc).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    best = {"train_acc": 0.0, "test_acc": 0.0, "loss": float("inf")}
    for ep in range(epochs):
        probe.train()
        loss = F.cross_entropy(probe(x_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            probe.eval()
            with torch.no_grad():
                tr_acc = (probe(x_tr).argmax(-1) == y_tr).float().mean().item()
                te_acc = (probe(x_te).argmax(-1) == y_te).float().mean().item()
            if loss.item() < best["loss"]:
                best = {"train_acc": tr_acc, "test_acc": te_acc, "loss": loss.item()}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", default=None, help="optional, only used to record metadata")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian mask σ in latent grid units")
    ap.add_argument("--probe_epochs", type=int, default=2000)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds_val = CachedLatentDataset(args.cache_dir, split="val",
                                 val_frac=args.val_frac, seed=args.seed)
    print(f"[data] val windows: {len(ds_val)}  (val_frac={args.val_frac})")

    # ---- camera sanity check on first 50 windows ----
    print("\n[sanity] camera projection check on first 50 windows…")
    sanity = []
    for i in range(min(50, len(ds_val))):
        b = ds_val[i]
        pos = b["positions"].numpy()
        vis = b["visibility"].numpy()
        sm = b["slot_mask"].numpy()
        for k in range(sm.shape[0]):
            if not bool(sm[k]):
                continue
            v = vis[:, k].astype(bool)
            if not v.any():
                continue
            pix = project_world_xy_to_pixel(pos[v, k])
            inside = ((pix >= 0) & (pix < IMG_SIZE)).all(axis=-1).mean()
            sanity.append(inside)
    sanity = np.array(sanity)
    print(f"[sanity] fraction-in-frame for visible (slot,frame) pairs: "
          f"mean={sanity.mean():.3f}  median={np.median(sanity):.3f}  "
          f"min={sanity.min():.3f}  N_slots={len(sanity)}")
    if sanity.mean() < 0.8:
        print("\n*** WARNING: camera projection looks off. Mean in-frame fraction "
              "for VISIBLE slots is < 80%. Continuing anyway, but results may "
              "not be reliable. ***\n")
    else:
        print("[sanity] projection looks reasonable.")

    # ---- gather features ----
    t0 = time.time()
    feats = gather(ds_val, args.batch_size, args.num_workers, args.sigma)
    n_videos = len({int(v.item()) for v in feats["video_id"]})
    print(f"[pool] {n_videos} val videos / {feats['feats'].shape[0]} windows "
          f"in {time.time()-t0:.1f}s")

    # ---- split by video ----
    vid_list = sorted({int(v.item()) for v in feats["video_id"]})
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(vid_list)
    half = len(vid_list) // 2
    tr_set, te_set = set(vid_list[:half]), set(vid_list[half:])
    print(f"[probe] {len(tr_set)} train videos / {len(te_set)} test videos")

    # ---- per-slot probes ----
    print(f"\n=== Per-slot raw Wan latent (Gaussian σ={args.sigma}, GT-position pool) ===")
    print(f"{'group':10s} {'classes':>8s} {'tr/te slots':>14s} {'chance':>8s} "
          f"{'majority':>10s} {'train_acc':>10s} {'TEST_ACC':>10s} {'Δ vs maj':>10s}")
    print("-" * 92)
    results = {}
    for gname, lo, hi, vocab in ATTR_GROUPS:
        x_tr, y_tr = per_slot_pairs(feats["feats"], feats["attrs"], feats["slot_mask"],
                                    feats["video_id"], tr_set, lo, hi)
        x_te, y_te = per_slot_pairs(feats["feats"], feats["attrs"], feats["slot_mask"],
                                    feats["video_id"], te_set, lo, hi)
        nc = hi - lo
        chance = 1.0 / nc
        maj = (torch.bincount(y_te, minlength=nc).max().item() / max(y_te.numel(), 1))
        best = train_linear(x_tr, y_tr, x_te, y_te, nc,
                            epochs=args.probe_epochs, lr=args.probe_lr,
                            wd=args.probe_wd, device=device)
        print(f"{gname:10s} {nc:>8d} {x_tr.shape[0]:>6d}/{x_te.shape[0]:<6d}  "
              f"{chance:>8.3f} {maj:>10.3f} {best['train_acc']:>10.3f} "
              f"{best['test_acc']:>10.3f} {best['test_acc']-maj:>+10.3f}")
        results[gname] = {"num_classes": nc, "majority": maj, **best,
                          "train_n": int(x_tr.shape[0]), "test_n": int(x_te.shape[0])}

    out = {
        "protocol": "Per-slot raw Wan-latent probe with GT-position Gaussian pool",
        "camera": {
            "pos": CAM_POS.tolist(), "target": CAM_TARGET.tolist(),
            "sensor_mm": SENSOR_MM, "focal_mm": FOCAL_MM, "obj_z": OBJ_Z,
            "img_size": IMG_SIZE,
        },
        "sigma_latent_units": args.sigma,
        "sanity_in_frame_mean": float(sanity.mean()),
        "sanity_in_frame_median": float(np.median(sanity)),
        "n_videos": n_videos,
        "n_windows": feats["feats"].shape[0],
        "results": results,
    }
    out_dir = Path(args.ckpt).parent if args.ckpt else Path(args.cache_dir).parent
    out_path = out_dir / "probe_wan_perslot_gtpos.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
