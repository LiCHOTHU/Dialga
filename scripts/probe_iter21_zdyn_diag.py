"""Iter 21 diagnostic: where is identity actually living?

Runs three linear identity probes off a single encoder pass over the val
split, distinguishing three hypotheses for why the per-video-averaged
z_static probe failed:

  z_static  per-video avg : the headline result from probe_iter21_identity
                            (= hypothesis 1 + 2 + 4 entangled)
  z_static  per-window    : if much higher than per-video avg, slot
                            assignment drifts across windows of the same
                            video (hypothesis 1).
  z_dyn_mean per-window   : z_dyn averaged over T → constant component
                            of z_dyn. If high, identity is sitting in
                            z_dyn rather than z_static (hypothesis 4).

Same protocol as probe_iter21_identity: trainer val split, 50/50 video
split for probe-train / probe-test, single Linear layer (no hidden),
50 evaluation points across 2000 epochs.

Usage:
    python scripts/probe_iter21_zdyn_diag.py \\
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
def gather_per_window_features(encoder, loader, device):
    """One encoder pass. Returns per-window arrays:
        video_ids : (N,) long
        z_static  : (N, K, d_static)        — per-window z_static
        z_dyn_mean: (N, K, d_dyn)           — z_dyn averaged over T
        attrs     : (N, K, A)
        slot_mask : (N, K) bool
    Also a per-video accumulator for the averaged-z_static reference.
    """
    vid_chunks, zs_chunks, zd_chunks, at_chunks, sm_chunks = [], [], [], [], []
    accum_avg = {}                                # for the avg reference
    for batch in loader:
        target = batch["latent"].to(device)
        B, C, T, H, W = target.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        z_static, z_dyn, _, _ = encoder(target, frame_mask=mask)
        z_static = z_static.float().cpu()                    # (B, K, d_static)
        z_dyn_mean = z_dyn.float().mean(dim=1).cpu()         # (B, K, d_dyn)
        vid_chunks.append(batch["video_id"].clone())
        zs_chunks.append(z_static)
        zd_chunks.append(z_dyn_mean)
        at_chunks.append(batch["attrs"].clone())
        sm_chunks.append(batch["slot_mask"].clone().bool())
        for b in range(B):
            vid = int(batch["video_id"][b].item())
            entry = accum_avg.setdefault(vid, {
                "sum": torch.zeros_like(z_static[b]),
                "n": 0,
                "attrs": batch["attrs"][b].clone(),
                "slot_mask": batch["slot_mask"][b].clone().bool(),
            })
            entry["sum"] += z_static[b]
            entry["n"] += 1
    out = {
        "video_id": torch.cat(vid_chunks),
        "z_static": torch.cat(zs_chunks),
        "z_dyn_mean": torch.cat(zd_chunks),
        "attrs": torch.cat(at_chunks),
        "slot_mask": torch.cat(sm_chunks),
    }
    per_video_avg = {
        vid: {
            "z_static": d["sum"] / max(d["n"], 1),
            "attrs": d["attrs"],
            "slot_mask": d["slot_mask"],
        }
        for vid, d in accum_avg.items()
    }
    return out, per_video_avg


def per_window_pairs(per_window, video_set, feature_key, group_lo, group_hi):
    """Build (x, y) over windows belonging to `video_set`."""
    vid = per_window["video_id"]
    feat = per_window[feature_key]            # (N, K, d)
    attrs = per_window["attrs"]               # (N, K, A)
    mask = per_window["slot_mask"]            # (N, K)
    keep = torch.tensor([int(v.item()) in video_set for v in vid], dtype=torch.bool)
    feat = feat[keep]
    attrs = attrs[keep]
    mask = mask[keep]
    xs, ys = [], []
    N, K = mask.shape
    for n in range(N):
        for k in range(K):
            if not bool(mask[n, k].item()):
                continue
            one_hot = attrs[n, k, group_lo:group_hi]
            if one_hot.sum().item() < 0.5:
                continue
            xs.append(feat[n, k])
            ys.append(int(one_hot.argmax().item()))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def per_video_pairs(per_video_avg, video_ids, group_lo, group_hi):
    xs, ys = [], []
    for vid in video_ids:
        d = per_video_avg[vid]
        attrs = d["attrs"]
        mask = d["slot_mask"]
        for k in range(attrs.shape[0]):
            if not bool(mask[k].item()):
                continue
            one_hot = attrs[k, group_lo:group_hi]
            if one_hot.sum().item() < 0.5:
                continue
            xs.append(d["z_static"][k])
            ys.append(int(one_hot.argmax().item()))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_linear_probe(x_tr, y_tr, x_te, y_te, num_classes,
                       epochs=2000, lr=1e-2, weight_decay=1e-4, device="cuda"):
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


def run_probe_set(name, pair_fn, video_args, ATTR_GROUPS, device, epochs, lr, wd):
    """Run the three (color/material/shape) probes for one feature variant."""
    out = {}
    print(f"\n=== {name} ===")
    print(f"{'group':10s} {'classes':>8s} {'tr/te slots':>14s} {'chance':>8s} "
          f"{'majority':>10s} {'train_acc':>10s} {'TEST_ACC':>10s} {'Δ vs maj':>10s}")
    print("-" * 92)
    for gname, lo, hi, vocab in ATTR_GROUPS:
        x_tr, y_tr = pair_fn(*video_args[0], lo, hi)
        x_te, y_te = pair_fn(*video_args[1], lo, hi)
        nc = hi - lo
        chance = 1.0 / nc
        maj = (torch.bincount(y_te, minlength=nc).max().item() / max(y_te.numel(), 1))
        best = train_linear_probe(x_tr, y_tr, x_te, y_te, nc,
                                  epochs=epochs, lr=lr, weight_decay=wd, device=device)
        print(f"{gname:10s} {nc:>8d} {x_tr.shape[0]:>6d}/{x_te.shape[0]:<6d}  "
              f"{chance:>8.3f} {maj:>10.3f} {best['train_acc']:>10.3f} "
              f"{best['test_acc']:>10.3f} {best['test_acc']-maj:>+10.3f}")
        out[gname] = {
            "num_classes": nc, "vocab": vocab,
            "n_train_slots": x_tr.shape[0], "n_test_slots": x_te.shape[0],
            "chance": chance, "majority_baseline": maj,
            "train_acc": best["train_acc"], "test_acc": best["test_acc"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
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
    print(f"[data] val windows: {len(ds_val)}  (val_frac={args.val_frac}, seed={args.seed})")

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
    per_window, per_video_avg = gather_per_window_features(enc, loader, device)
    n_videos = len(per_video_avg)
    print(f"[encode] {n_videos} val videos / {per_window['z_static'].shape[0]} windows "
          f"in {time.time()-t0:.1f}s")

    vid_list = sorted(per_video_avg.keys())
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(vid_list)
    half = len(vid_list) // 2
    tr_vids = vid_list[:half]
    te_vids = vid_list[half:]
    tr_set, te_set = set(tr_vids), set(te_vids)
    print(f"[probe] {len(tr_vids)} probe-train videos / {len(te_vids)} probe-test videos "
          f"(seed={args.probe_split_seed})")

    # Feature variants:
    results = {}
    results["z_static_per_video_avg"] = run_probe_set(
        "z_static  (per-video avg, K=8 × 16)",
        per_video_pairs,
        [(per_video_avg, tr_vids), (per_video_avg, te_vids)],
        ATTR_GROUPS, device, args.probe_epochs, args.probe_lr, args.probe_wd,
    )
    results["z_static_per_window"] = run_probe_set(
        "z_static  (per-window, K=8 × 16)",
        lambda *aa: per_window_pairs(per_window, *aa),
        [(tr_set, "z_static"), (te_set, "z_static")],
        ATTR_GROUPS, device, args.probe_epochs, args.probe_lr, args.probe_wd,
    )
    results["z_dyn_mean_per_window"] = run_probe_set(
        "z_dyn_mean (per-window, K=8 × 32)",
        lambda *aa: per_window_pairs(per_window, *aa),
        [(tr_set, "z_dyn_mean"), (te_set, "z_dyn_mean")],
        ATTR_GROUPS, device, args.probe_epochs, args.probe_lr, args.probe_wd,
    )

    print("\n=== summary (test accuracy / majority baseline) ===")
    print(f"{'group':10s} | {'z_static avg':>14s} | {'z_static win':>14s} | {'z_dyn_mean':>14s} | {'majority':>10s}")
    for g, _, _, _ in ATTR_GROUPS:
        rs_avg = results["z_static_per_video_avg"][g]
        rs_win = results["z_static_per_window"][g]
        rd_win = results["z_dyn_mean_per_window"][g]
        print(f"{g:10s} | {rs_avg['test_acc']:>14.3f} | {rs_win['test_acc']:>14.3f} | "
              f"{rd_win['test_acc']:>14.3f} | {rs_avg['majority_baseline']:>10.3f}")

    out_path = Path(args.ckpt).parent / "probe_identity_diag.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": "linear-logistic; 50/50 video split within trainer val pool; "
                        "three feature variants",
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
