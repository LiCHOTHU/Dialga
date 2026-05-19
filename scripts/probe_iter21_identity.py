"""Iter 21 identity probe (linear): does z_static encode color/material/shape?

Protocol (held-out, linear-only):
  1. Load trained TrajectoryEncoder checkpoint.
  2. Use ONLY the trainer's val split (val_frac=0.2, same seed=42) — the
     encoder never saw these videos during training.
  3. Encode each val video → z_static of shape (K=8, 16), averaged across
     the 4 windows of that video.
  4. For each (video, slot) where GT slot_mask=1, pair z_static[n, k] with
     the CLEVRER attrs (color 8-way, material 2-way, shape 3-way).
  5. Split 50/50 by video_id (NOT by slot) so probe-train and probe-test
     videos are disjoint.
  6. Train ONE linear layer per attribute group (no hidden layer — the
     probe is kept weak so the result speaks to the representation, not
     the probe).
  7. Report test accuracy vs the 12.5% / 50% / 33% chance rates and the
     empirical majority-class baseline.

Usage:
    python scripts/probe_iter21_identity.py \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W12 \\
        --ckpt outputs/iter21_10000vid_<stamp>/trajectory.pt
"""
from __future__ import annotations

import argparse
import json
import random
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
from scripts.train_trajectory import CachedLatentDataset, collate


ATTR_GROUPS = [
    ("color",    0,                                           len(COLOR_VOCAB),                                          COLOR_VOCAB),
    ("material", len(COLOR_VOCAB),                            len(COLOR_VOCAB) + len(MATERIAL_VOCAB),                    MATERIAL_VOCAB),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB),      len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB), SHAPE_VOCAB),
]


@torch.no_grad()
def compute_z_static_per_video(encoder, loader, device):
    """Encode windows → z_static, average across the windows of each video.

    Returns dict[video_id] = {
        "z_static":  (K, d_static),   averaged across windows
        "attrs":     (K, A),          one-hot block over color/material/shape
        "slot_mask": (K,) bool,
    }
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


def slot_pairs(per_video, video_ids, group_lo, group_hi):
    """Flatten to per-(video, slot) (x, y) pairs across a given video list."""
    xs, ys = [], []
    for vid in video_ids:
        d = per_video[vid]
        attrs = d["attrs"]                          # (K, A)
        mask = d["slot_mask"]                       # (K,)
        for k in range(attrs.shape[0]):
            if not bool(mask[k].item()):
                continue
            one_hot = attrs[k, group_lo:group_hi]
            if one_hot.sum().item() < 0.5:          # padded / unused
                continue
            xs.append(d["z_static"][k])
            ys.append(int(one_hot.argmax().item()))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_linear_probe(x_tr, y_tr, x_te, y_te, num_classes,
                       epochs=2000, lr=1e-2, weight_decay=1e-4, device="cuda"):
    """Single Linear layer (logistic regression). Trained full-batch."""
    d_in = x_tr.shape[1]
    probe = nn.Linear(d_in, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    best = {"train_acc": 0.0, "test_acc": 0.0, "ep": 0, "loss": float("inf")}
    for ep in range(epochs):
        probe.train()
        logits = probe(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            probe.eval()
            with torch.no_grad():
                tr_acc = (probe(x_tr).argmax(-1) == y_tr).float().mean().item()
                te_acc = (probe(x_te).argmax(-1) == y_te).float().mean().item()
            if loss.item() < best["loss"]:
                best = {"train_acc": tr_acc, "test_acc": te_acc, "ep": ep, "loss": loss.item()}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0,
                    help="Seed for the 50/50 probe-train / probe-test video split.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--probe_epochs", type=int, default=2000)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds_val = CachedLatentDataset(args.cache_dir, split="val",
                                 val_frac=args.val_frac, seed=args.seed)
    print(f"[data] val windows: {len(ds_val)}  (from val_frac={args.val_frac}, seed={args.seed})")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    K        = int(a.get("K", 8))
    d_model  = int(a.get("d_model", 192))
    n_heads  = int(a.get("n_heads", 4))
    n_layers = int(a.get("n_layers", 4))
    d_static = int(a.get("d_static", 16))
    d_dyn    = int(a.get("d_dyn", 32))

    sample = ds_val[0]
    C, T, H, W = sample["latent"].shape
    enc = TrajectoryEncoder(
        latent_ch=C, K=K, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        T_max=max(T * 2, 16), spatial_size=H, d_static=d_static, d_dyn=d_dyn,
        dropout=0.0,
    ).to(device)
    enc.load_state_dict(ckpt.get("encoder_state_dict", ckpt.get("encoder", ckpt)))
    enc.eval()
    print(f"[model] encoder loaded  K={K} d_static={d_static} d_dyn={d_dyn}")

    loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    t0 = time.time()
    per_video = compute_z_static_per_video(enc, loader, device)
    n_videos = len(per_video)
    print(f"[encode] {n_videos} val videos in {time.time()-t0:.1f}s")

    vid_list = sorted(per_video.keys())
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(vid_list)
    half = len(vid_list) // 2
    tr_vids = vid_list[:half]
    te_vids = vid_list[half:]
    print(f"[probe] split: {len(tr_vids)} probe-train videos / {len(te_vids)} probe-test videos "
          f"(seed={args.probe_split_seed})")

    results = {}
    print(f"\n{'group':10s} {'classes':>8s} {'tr/te slots':>14s} {'chance':>8s} "
          f"{'majority':>10s} {'train_acc':>10s} {'TEST_ACC':>10s} {'Δ vs maj':>10s}")
    print("-" * 92)
    for name, lo, hi, vocab in ATTR_GROUPS:
        x_tr, y_tr = slot_pairs(per_video, tr_vids, lo, hi)
        x_te, y_te = slot_pairs(per_video, te_vids, lo, hi)
        num_classes = hi - lo
        chance = 1.0 / num_classes
        majority = (torch.bincount(y_te, minlength=num_classes).max().item()
                    / max(y_te.numel(), 1))
        best = train_linear_probe(
            x_tr, y_tr, x_te, y_te, num_classes,
            epochs=args.probe_epochs, lr=args.probe_lr,
            weight_decay=args.probe_wd, device=device,
        )
        print(f"{name:10s} {num_classes:>8d} "
              f"{x_tr.shape[0]:>6d}/{x_te.shape[0]:<6d}  "
              f"{chance:>8.3f} {majority:>10.3f} "
              f"{best['train_acc']:>10.3f} {best['test_acc']:>10.3f} "
              f"{best['test_acc']-majority:>+10.3f}")
        results[name] = {
            "num_classes": num_classes,
            "vocab": vocab,
            "n_train_slots": x_tr.shape[0],
            "n_test_slots": x_te.shape[0],
            "chance": chance,
            "majority_baseline": majority,
            "train_acc": best["train_acc"],
            "test_acc": best["test_acc"],
            "best_ep": best["ep"],
        }

    out_path = Path(args.ckpt).parent / "probe_identity_linear.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": "linear-logistic, 50/50 video split within trainer val pool",
            "cache_dir": args.cache_dir,
            "ckpt": args.ckpt,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "probe_split_seed": args.probe_split_seed,
            "n_val_videos": n_videos,
            "n_probe_train_videos": len(tr_vids),
            "n_probe_test_videos": len(te_vids),
            "results": results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
