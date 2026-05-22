"""v5 diagnostic: where is identity actually living?

Two diagnostics in one pass over the val split, against a v5 checkpoint:

  (A) z_static collapse stats
      Per-dim std of z_static across val videos. If most dims have
      std near 0, the encoder is producing a near-constant z_static
      (L_consist MSE has been satisfied by collapse, not by encoding
      identity).

  (B) z_dyn identity probes
      Same scene-level modal-attribute linear probe as probe_v5_modal,
      but on z_dyn instead of z_static. Two variants:
        z_dyn_enc   : encoder output, averaged over (windows, L_obs)
        z_dyn_roll  : rolled state at the LAST predicted frame,
                      averaged over windows.

      If z_dyn_enc probes well above majority but z_static doesn't,
      identity is sitting in z_dyn (the W_proj bottleneck did not
      bite).
      If z_dyn_roll also probes well above majority, identity is
      surviving the bottleneck *through rollout* — confirming the
      "vel half preserved under zero-init accel" leak.

Usage:
    python scripts/probe_v5_zdyn_diag.py \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W12 \\
        --ckpt outputs/v5_phase5_<stamp>/v5.pt
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
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_encoder import LatentEncoder3D
from scripts.legacy.train_trajectory import CachedLatentDataset, collate


ATTR_GROUPS = [
    ("color",    0,                                      len(COLOR_VOCAB)),
    ("material", len(COLOR_VOCAB),                       len(COLOR_VOCAB) + len(MATERIAL_VOCAB)),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB), len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB)),
]


def modal_label(attrs, slot_mask, lo, hi):
    real_k = slot_mask.bool()
    if not real_k.any():
        return -1
    block = attrs[real_k, lo:hi]
    classes = block.argmax(dim=-1)
    counts = torch.bincount(classes, minlength=hi - lo)
    return int(counts.argmax().item())


@torch.no_grad()
def compute_features_per_video(encoder, fwd, loader, L_obs, L_pred, device):
    """One pass. Returns per-video averaged features.

    v5.1 semantics: encoder returns dict with z_static (B, D_s), z_dyn (B, D_d)
    — no time axis on z_dyn. We compute z_dyn_roll by stepping `fwd` `L_pred`
    times from the encoded z_dyn.
    """
    accum = {}
    for batch in loader:
        latent = batch["latent"].to(device)              # (B, C, T_lat, H, W)
        out = encoder(latent)
        if isinstance(out, dict):
            z_static, z_dyn = out["z_static"], out["z_dyn"]
        else:
            z_static, z_dyn = out
        # collapse per-frame z_dyn -> chunk vector for the probe
        if z_dyn.dim() == 3:
            z_dyn = z_dyn.mean(dim=1)
        z_roll_last = z_dyn
        for _ in range(max(L_pred, 1)):
            z_roll_last = fwd.step(z_roll_last)
        z_static = z_static.float().cpu()
        z_dyn_enc_mean = z_dyn.float().cpu()
        z_roll_last = z_roll_last.float().cpu()
        for b in range(latent.shape[0]):
            vid = int(batch["video_id"][b].item())
            e = accum.setdefault(vid, {
                "s_sum": torch.zeros_like(z_static[b]),
                "de_sum": torch.zeros_like(z_dyn_enc_mean[b]),
                "dr_sum": torch.zeros_like(z_roll_last[b]),
                "n": 0,
                "attrs": batch["attrs"][b].clone(),
                "slot_mask": batch["slot_mask"][b].clone().bool(),
            })
            e["s_sum"] += z_static[b]
            e["de_sum"] += z_dyn_enc_mean[b]
            e["dr_sum"] += z_roll_last[b]
            e["n"] += 1
    out = {}
    for vid, d in accum.items():
        n = max(d["n"], 1)
        out[vid] = {
            "z_static": d["s_sum"] / n,
            "z_dyn_enc": d["de_sum"] / n,
            "z_dyn_roll": d["dr_sum"] / n,
            "attrs": d["attrs"],
            "slot_mask": d["slot_mask"],
        }
    return out


def build_xy(per_video, video_ids, lo, hi, key):
    xs, ys = [], []
    for vid in video_ids:
        d = per_video[vid]
        y = modal_label(d["attrs"], d["slot_mask"], lo, hi)
        if y < 0:
            continue
        xs.append(d[key])
        ys.append(y)
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


def collapse_stats(per_video, key):
    """Per-dim std across videos for a given feature key."""
    X = torch.stack([d[key] for d in per_video.values()], dim=0)  # (N, D)
    std_per_dim = X.std(dim=0)                                     # (D,)
    mean_per_dim = X.mean(dim=0)
    return {
        "n_videos": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "mean_std": float(std_per_dim.mean().item()),
        "min_std": float(std_per_dim.min().item()),
        "max_std": float(std_per_dim.max().item()),
        "n_dims_below_0.01": int((std_per_dim < 0.01).sum().item()),
        "n_dims_below_0.1": int((std_per_dim < 0.1).sum().item()),
        "std_per_dim": std_per_dim.tolist(),
        "mean_per_dim_abs": float(mean_per_dim.abs().mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
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
    L_obs = int(a.get("L_obs", 1))
    L_pred = int(a.get("L_pred", 2))
    d_static = int(a.get("d_static", 32))
    d_dyn = int(a.get("d_dyn", 16))
    d_state = int(a.get("d_state", 8))
    enc_hidden_ch = int(a.get("enc_hidden_ch", a.get("hidden_ch", 64)))
    no_proj = bool(a.get("no_proj", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    fwd.load_state_dict(ckpt["fwd"])
    fwd.eval()
    print(f"[model] L_obs={L_obs} L_pred={L_pred} d_static={d_static} d_dyn={d_dyn} d_state={d_state} enc_hidden={enc_hidden_ch}")

    loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    t0 = time.time()
    per_video = compute_features_per_video(enc, fwd, loader, L_obs, L_pred, device)
    n_videos = len(per_video)
    print(f"[encode] {n_videos} val videos in {time.time()-t0:.1f}s")

    # ----- (A) collapse stats -----
    print("\n========== (A) Collapse stats — per-dim std across videos ==========")
    collapse = {}
    for key in ("z_static", "z_dyn_enc", "z_dyn_roll"):
        s = collapse_stats(per_video, key)
        collapse[key] = s
        print(f"{key:12s} dim={s['dim']:3d} | "
              f"mean_std={s['mean_std']:.4f} | min={s['min_std']:.4f} "
              f"max={s['max_std']:.4f} | mean|μ|={s['mean_per_dim_abs']:.3f} | "
              f"dims<0.01: {s['n_dims_below_0.01']:2d}  dims<0.1: {s['n_dims_below_0.1']:2d}")

    # ----- (B) identity probes -----
    vid_list = sorted(per_video.keys())
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(vid_list)
    half = len(vid_list) // 2
    tr_vids, te_vids = vid_list[:half], vid_list[half:]

    print("\n========== (B) Identity probes (linear, 50/50 video split) ==========")
    print(f"{'feature':12s} {'group':10s} {'tr/te':>12s} "
          f"{'majority':>10s} {'train':>10s} {'TEST':>10s} {'Δ':>10s}")
    print("-" * 86)
    probe_results = {}
    for feat_key in ("z_static", "z_dyn_enc", "z_dyn_roll"):
        probe_results[feat_key] = {}
        for name, lo, hi in ATTR_GROUPS:
            x_tr, y_tr = build_xy(per_video, tr_vids, lo, hi, feat_key)
            x_te, y_te = build_xy(per_video, te_vids, lo, hi, feat_key)
            num_classes = hi - lo
            majority = (torch.bincount(y_te, minlength=num_classes).max().item()
                        / max(y_te.numel(), 1))
            best = train_linear_probe(
                x_tr, y_tr, x_te, y_te, num_classes,
                epochs=args.probe_epochs, lr=args.probe_lr,
                weight_decay=args.probe_wd, device=device,
            )
            delta = best["test_acc"] - majority
            print(f"{feat_key:12s} {name:10s} "
                  f"{x_tr.shape[0]:>5d}/{x_te.shape[0]:<5d}  "
                  f"{majority:>10.3f} "
                  f"{best['train_acc']:>10.3f} {best['test_acc']:>10.3f} "
                  f"{delta:>+10.3f}")
            probe_results[feat_key][name] = {
                "num_classes": num_classes,
                "majority": majority,
                "train_acc": best["train_acc"],
                "test_acc": best["test_acc"],
                "delta_vs_majority": delta,
            }

    out_path = Path(args.ckpt).parent / "probe_v5_zdyn_diag.json"
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "L_obs": L_obs, "L_pred": L_pred,
            "n_val_videos": n_videos,
            "n_probe_train_videos": len(tr_vids),
            "n_probe_test_videos": len(te_vids),
            "collapse": collapse,
            "probes": probe_results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
