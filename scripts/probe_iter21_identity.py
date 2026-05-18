"""Iter 21 identity probe: does z_static encode color/material/shape?

Protocol:
  1. Load the trained TrajectoryEncoder checkpoint.
  2. Re-split the cached windows the same way training did (val_frac=0.2 by video_id).
  3. Forward each window through the encoder → z_static[B, K, 16].
  4. Average z_static across the 4 windows of each video → one z_static[K, 16] per video.
  5. Mask down to real slots (slot_mask=True). Each (video, slot) is one (x, y) pair where
     x = z_static and y = (color, material, shape) one-hot labels from attrs.
  6. Train an MLP probe on TRAIN-video (z_static[k], attrs[k]) pairs; report VAL accuracy
     for each of the three attribute types, plus the chance baseline.

A z_static dimension that genuinely carries identity should let the probe beat chance on
val. If val accuracy is at chance, the encoder has memorized episode-specific surface
features rather than learned a transferable mapping. That is the load-bearing test of
the disentanglement claim.

Usage:
    python scripts/probe_iter21_identity.py \\
        --cache_dir outputs/cache/wan_10000vid_W12 \\
        --ckpt outputs/iter21_10000vid_<stamp>/trajectory.pt \\
        --val_frac 0.2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.model.trajectory_encoder import TrajectoryEncoder

# Re-use the cache dataset + collate logic from training.
from scripts.train_trajectory import CachedLatentDataset, collate


ATTR_GROUPS = [
    ("color", 0, len(COLOR_VOCAB), COLOR_VOCAB),
    ("material", len(COLOR_VOCAB), len(COLOR_VOCAB) + len(MATERIAL_VOCAB), MATERIAL_VOCAB),
    ("shape",
     len(COLOR_VOCAB) + len(MATERIAL_VOCAB),
     len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB),
     SHAPE_VOCAB),
]


@torch.no_grad()
def compute_z_static_per_video(encoder, loader, device):
    """Average encoder z_static across all windows of each video.

    Returns:
        per_video: dict[video_id] = {"z_static": (K, d_static), "attrs": (K, A), "slot_mask": (K,)}
    """
    accum = {}
    for batch in loader:
        target = batch["latent"].to(device)
        B, C, T, H, W = target.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        z_static, _, _, _ = encoder(target, frame_mask=mask)
        z_static = z_static.float().cpu()
        for b in range(B):
            vid = int(batch["video_id"][b].item())
            entry = accum.setdefault(vid, {
                "sum": torch.zeros_like(z_static[b]),
                "n": 0,
                "attrs": batch["attrs"][b].clone(),
                "slot_mask": batch["slot_mask"][b].clone().bool(),
            })
            entry["sum"] += z_static[b]
            entry["n"] += 1
    return {
        vid: {
            "z_static": d["sum"] / max(d["n"], 1),
            "attrs": d["attrs"],
            "slot_mask": d["slot_mask"],
        }
        for vid, d in accum.items()
    }


def slot_dataset(per_video, group_lo, group_hi):
    """Flatten to per-(video, slot) (x, y_label) pairs, filtered by slot_mask."""
    xs, ys = [], []
    for vid, d in per_video.items():
        attrs = d["attrs"]                              # (K, A)
        mask = d["slot_mask"]                           # (K,)
        for k in range(attrs.shape[0]):
            if not bool(mask[k].item()):
                continue
            one_hot = attrs[k, group_lo:group_hi]
            if one_hot.sum().item() < 0.5:              # unused / padded
                continue
            xs.append(d["z_static"][k])
            ys.append(int(one_hot.argmax().item()))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_probe(x_train, y_train, x_val, y_val, num_classes,
                hidden=64, epochs=200, lr=1e-3, weight_decay=1e-4,
                device="cuda", verbose=False):
    """Train a 2-layer MLP probe on (x, y). Returns val accuracy at best train loss."""
    d_in = x_train.shape[1]
    head = nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, num_classes),
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    x_train = x_train.to(device); y_train = y_train.to(device)
    x_val = x_val.to(device); y_val = y_val.to(device)
    best_val_acc = 0.0
    best_train_acc = 0.0
    for ep in range(epochs):
        head.train()
        logits = head(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                train_acc = (head(x_train).argmax(-1) == y_train).float().mean().item()
                val_acc = (head(x_val).argmax(-1) == y_val).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
            if verbose and ep % 50 == 0:
                print(f"   ep {ep:3d}  loss={loss.item():.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
    return best_train_acc, best_val_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--probe_hidden", type=int, default=64)
    ap.add_argument("--probe_epochs", type=int, default=400)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Re-create the same split the trainer used.
    ds_train = CachedLatentDataset(args.cache_dir, split="train",
                                   val_frac=args.val_frac, seed=args.seed)
    ds_val = CachedLatentDataset(args.cache_dir, split="val",
                                 val_frac=args.val_frac, seed=args.seed)
    print(f"[data] train={len(ds_train)} windows, val={len(ds_val)} windows")
    sample = ds_train[0]
    C, T, H, W = sample["latent"].shape
    print(f"[data] latent shape: ({C}, {T}, {H}, {W})")

    # Load ckpt + read its hparams (saved by train_trajectory.py).
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "args" in ckpt:
        a = ckpt["args"]
        K = int(a.get("K", 8))
        d_model = int(a.get("d_model", 192))
        n_heads = int(a.get("n_heads", 4))
        n_layers = int(a.get("n_layers", 4))
        d_static = int(a.get("d_static", 16))
        d_dyn = int(a.get("d_dyn", 32))
    else:
        K, d_model, n_heads, n_layers, d_static, d_dyn = 8, 192, 4, 4, 16, 32

    enc = TrajectoryEncoder(
        latent_ch=C, K=K, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        T_max=max(T * 2, 16), spatial_size=H, d_static=d_static, d_dyn=d_dyn,
        dropout=0.0,
    ).to(device)
    state_dict = ckpt.get("encoder_state_dict", ckpt.get("encoder", ckpt))
    enc.load_state_dict(state_dict)
    enc.eval()
    print(f"[model] encoder loaded; d_static={d_static}, K={K}")

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    print("[encode] computing z_static averaged across windows per video…")
    t0 = time.time()
    train_pv = compute_z_static_per_video(enc, loader_train, device)
    print(f"   train: {len(train_pv)} videos in {time.time()-t0:.1f}s")
    t0 = time.time()
    val_pv = compute_z_static_per_video(enc, loader_val, device)
    print(f"   val:   {len(val_pv)} videos in {time.time()-t0:.1f}s")

    results = {}
    for name, lo, hi, vocab in ATTR_GROUPS:
        x_tr, y_tr = slot_dataset(train_pv, lo, hi)
        x_va, y_va = slot_dataset(val_pv, lo, hi)
        num_classes = hi - lo
        chance = 1.0 / num_classes
        # Empirical majority-class baseline on val.
        if y_va.numel() > 0:
            counts = torch.bincount(y_va, minlength=num_classes)
            majority = counts.max().item() / max(y_va.numel(), 1)
        else:
            majority = chance
        print(f"\n=== probe: {name} ({num_classes} classes, vocab={vocab}) ===")
        print(f"  train slots: {x_tr.shape[0]}  val slots: {x_va.shape[0]}")
        print(f"  chance: {chance:.3f}  majority-baseline (val): {majority:.3f}")
        if x_tr.shape[0] < 10 or x_va.shape[0] < 10:
            print("  [skip] too few examples")
            continue
        train_acc, val_acc = train_probe(
            x_tr, y_tr, x_va, y_va, num_classes,
            hidden=args.probe_hidden, epochs=args.probe_epochs,
            lr=args.probe_lr, device=device,
        )
        print(f"  → train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
              f"(margin over chance: +{val_acc - chance:+.3f}, "
              f"over majority: {val_acc - majority:+.3f})")
        results[name] = {
            "num_classes": num_classes,
            "vocab": vocab,
            "n_train_slots": x_tr.shape[0],
            "n_val_slots": x_va.shape[0],
            "chance": chance,
            "majority_baseline": majority,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

    print("\n=== summary ===")
    for name, r in results.items():
        print(f"  {name:8s}  val={r['val_acc']:.3f}  "
              f"chance={r['chance']:.3f}  maj={r['majority_baseline']:.3f}  "
              f"(margin: {r['val_acc'] - r['majority_baseline']:+.3f})")

    out_path = Path(args.ckpt).parent / "probe_identity.json"
    with open(out_path, "w") as f:
        json.dump({
            "cache_dir": args.cache_dir,
            "ckpt": args.ckpt,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "results": results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
