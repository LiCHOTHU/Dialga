"""Experiment 1 — falsification test for "identity is in the Wan latent".

Two probes, both linear, both with train/test reported:

  (A) RAW WAN LATENT, global pool:
      x = latent.mean(dim=(T,H,W))            shape (N, 48)
      y = multi-label "any visible slot has color/material/shape c"
      Trains one binary linear probe per class. Reports mean test acc.
      Tests: is the cue ANYWHERE in the Wan latent?

  (B) ENCODER PRE-PROJECTION static_out / frame_slot_out:
      Same per-slot protocol as probe_iter21_zdyn_diag, but read the
      192-d tokens BEFORE static_head / dyn_head. Tells us whether the
      d_model→d_static bottleneck is destroying signal that the encoder
      body actually holds.

Run:
    python scripts/probe_wan_latent.py \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W12 \\
        --ckpt outputs/iter22c_attrs_20260519_030044/trajectory.pt
"""
from __future__ import annotations

import argparse
import json
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
from src.model.trajectory_encoder_v21 import TrajectoryEncoder
from scripts.legacy.train_trajectory import CachedLatentDataset, collate


ATTR_GROUPS = [
    ("color",    0,                                       len(COLOR_VOCAB),                                          COLOR_VOCAB),
    ("material", len(COLOR_VOCAB),                        len(COLOR_VOCAB) + len(MATERIAL_VOCAB),                    MATERIAL_VOCAB),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB),  len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB), SHAPE_VOCAB),
]


@torch.no_grad()
def gather_features(encoder, loader, device):
    """One pass: gather Wan latents (CPU), pre-projection 192-d static + dyn tokens.

    Returns dict with:
        video_id      : (N,)
        latent_global : (N, 48)    — global mean over (T,H,W)
        static_192    : (N, K, 192)
        frame_192_mean: (N, K, 192)— frame_slot_out.mean(dim=T)
        attrs         : (N, K, 13)
        slot_mask     : (N, K) bool
    """
    vids, latent_globals, st192, fr192, attrs_all, smasks = [], [], [], [], [], []
    for batch in loader:
        target = batch["latent"].to(device)             # (B, 48, T_lat, 8, 8)
        B, C, T, H, W = target.shape

        # ----- (A) raw latent global pool -----
        latent_globals.append(target.float().mean(dim=(2, 3, 4)).cpu())

        # ----- (B) pre-projection encoder features -----
        # Re-run the encoder body manually so we can intercept tokens BEFORE
        # static_head / dyn_head. This mirrors TrajectoryEncoder.forward exactly
        # up to the projection.
        e = encoder
        S = H * W
        x = target.permute(0, 2, 3, 4, 1).reshape(B, T, S, C)
        x = e.input_proj(x)
        x = x + e.spatial_pos
        t_idx = torch.arange(T, device=x.device)
        x = x + e.temporal_pos(t_idx)[None, :, None, :]

        static_q = e.slot_queries.expand(B, -1, -1)
        frame_q = e.frame_slot_queries.expand(B, T, -1, -1)
        frame_q = frame_q + e.temporal_pos(t_idx)[None, :, None, :]

        patch_tokens = x.reshape(B, T * S, -1)
        frame_slot_tokens = frame_q.reshape(B, T * e.K, -1)
        tokens = torch.cat([static_q, frame_slot_tokens, patch_tokens], dim=1)
        tokens = e.encoder(tokens)
        tokens = e.out_norm(tokens)
        static_out = tokens[:, :e.K]                                       # (B, K, d_model)
        frame_slot_out = tokens[:, e.K:e.K + T * e.K].reshape(B, T, e.K, -1)
        frame_mean = frame_slot_out.mean(dim=1)                            # (B, K, d_model)

        st192.append(static_out.float().cpu())
        fr192.append(frame_mean.float().cpu())

        vids.append(batch["video_id"].clone())
        attrs_all.append(batch["attrs"].clone())
        smasks.append(batch["slot_mask"].clone().bool())
    return {
        "video_id":       torch.cat(vids),
        "latent_global":  torch.cat(latent_globals),
        "static_192":     torch.cat(st192),
        "frame_192_mean": torch.cat(fr192),
        "attrs":          torch.cat(attrs_all),
        "slot_mask":      torch.cat(smasks),
    }


def split_by_video(video_ids, probe_split_seed):
    vid_list = sorted({int(v.item()) for v in video_ids})
    rng = random.Random(probe_split_seed)
    rng.shuffle(vid_list)
    half = len(vid_list) // 2
    return set(vid_list[:half]), set(vid_list[half:])


def per_slot_pairs(feat, attrs, slot_mask, video_ids, video_set, lo, hi):
    """(N,K,d) → (slot_count, d) features and (slot_count,) class labels."""
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


def train_linear(x_tr, y_tr, x_te, y_te, num_classes, epochs=2000, lr=1e-2, wd=1e-4, device="cuda"):
    probe = nn.Linear(x_tr.shape[1], num_classes).to(device)
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


def build_multilabel(feats, attrs, slot_mask, video_ids, video_set, lo, hi):
    """Window-level multi-label. y[n, c] = 1 iff any visible slot in window n has class c.
    Returns x (N_win, d), y (N_win, n_classes) float.
    """
    keep_win = torch.tensor([int(v.item()) in video_set for v in video_ids], dtype=torch.bool)
    feats = feats[keep_win]; attrs_ = attrs[keep_win]; mask = slot_mask[keep_win]
    nc = hi - lo
    N = feats.shape[0]
    y = torch.zeros(N, nc, dtype=torch.float32)
    for n in range(N):
        for k in range(mask.shape[1]):
            if not bool(mask[n, k].item()):
                continue
            oh = attrs_[n, k, lo:hi]
            if oh.sum().item() < 0.5:
                continue
            y[n, int(oh.argmax().item())] = 1.0
    return feats, y


def train_multilabel_linear(x_tr, y_tr, x_te, y_te, epochs=2000, lr=1e-2, wd=1e-4, device="cuda"):
    """One binary head per class. Reports mean per-class accuracy and pos-rate."""
    nc = y_tr.shape[1]
    probe = nn.Linear(x_tr.shape[1], nc).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    pos_rate_te = y_te.mean(dim=0).cpu().numpy()
    best = {"train_acc": 0.0, "test_acc": 0.0, "loss": float("inf"),
            "per_class_test": [0.0] * nc}
    for ep in range(epochs):
        probe.train()
        logits = probe(x_tr)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            probe.eval()
            with torch.no_grad():
                tr_pred = (probe(x_tr) > 0).float()
                te_pred = (probe(x_te) > 0).float()
                tr_acc = (tr_pred == y_tr).float().mean().item()
                te_acc = (te_pred == y_te).float().mean().item()
                pc = ((te_pred == y_te).float().mean(dim=0)).cpu().numpy().tolist()
            if loss.item() < best["loss"]:
                best = {"train_acc": tr_acc, "test_acc": te_acc, "loss": loss.item(),
                        "per_class_test": pc}
    return best, pos_rate_te.tolist()


def run_per_slot(name, feats_NKd, attrs, slot_mask, video_ids, tr_set, te_set,
                 device, epochs, lr, wd):
    print(f"\n=== {name} ===")
    print(f"{'group':10s} {'classes':>8s} {'tr/te slots':>14s} {'chance':>8s} "
          f"{'majority':>10s} {'train_acc':>10s} {'TEST_ACC':>10s} {'Δ vs maj':>10s}")
    print("-" * 92)
    out = {}
    for gname, lo, hi, vocab in ATTR_GROUPS:
        x_tr, y_tr = per_slot_pairs(feats_NKd, attrs, slot_mask, video_ids, tr_set, lo, hi)
        x_te, y_te = per_slot_pairs(feats_NKd, attrs, slot_mask, video_ids, te_set, lo, hi)
        nc = hi - lo
        chance = 1.0 / nc
        maj = (torch.bincount(y_te, minlength=nc).max().item() / max(y_te.numel(), 1))
        best = train_linear(x_tr, y_tr, x_te, y_te, nc, epochs=epochs, lr=lr, wd=wd, device=device)
        print(f"{gname:10s} {nc:>8d} {x_tr.shape[0]:>6d}/{x_te.shape[0]:<6d}  "
              f"{chance:>8.3f} {maj:>10.3f} {best['train_acc']:>10.3f} "
              f"{best['test_acc']:>10.3f} {best['test_acc']-maj:>+10.3f}")
        out[gname] = {"num_classes": nc, "majority": maj,
                      "train_acc": best["train_acc"], "test_acc": best["test_acc"]}
    return out


def run_multilabel(name, feats_Nd, attrs, slot_mask, video_ids, tr_set, te_set,
                   device, epochs, lr, wd):
    print(f"\n=== {name} (multi-label: any-visible-slot-has-class) ===")
    print(f"{'group':10s} {'classes':>8s} {'tr/te wins':>14s} {'pos rate':>10s} "
          f"{'always-0':>10s} {'train_acc':>10s} {'TEST_ACC':>10s} {'Δ vs all-0':>12s}")
    print("-" * 100)
    out = {}
    for gname, lo, hi, vocab in ATTR_GROUPS:
        x_tr, y_tr = build_multilabel(feats_Nd, attrs, slot_mask, video_ids, tr_set, lo, hi)
        x_te, y_te = build_multilabel(feats_Nd, attrs, slot_mask, video_ids, te_set, lo, hi)
        nc = hi - lo
        best, pos_rate = train_multilabel_linear(x_tr, y_tr, x_te, y_te,
                                                 epochs=epochs, lr=lr, wd=wd, device=device)
        # "always-0" baseline mean acc = mean(1 - pos_rate)
        all0 = float(np.mean([1.0 - p for p in pos_rate]))
        avg_pos = float(np.mean(pos_rate))
        print(f"{gname:10s} {nc:>8d} {x_tr.shape[0]:>6d}/{x_te.shape[0]:<6d}  "
              f"{avg_pos:>10.3f} {all0:>10.3f} {best['train_acc']:>10.3f} "
              f"{best['test_acc']:>10.3f} {best['test_acc']-all0:>+12.3f}")
        out[gname] = {"num_classes": nc, "avg_pos_rate": avg_pos,
                      "always_zero_baseline": all0,
                      "train_acc": best["train_acc"], "test_acc": best["test_acc"],
                      "per_class_test_acc": best["per_class_test"],
                      "per_class_pos_rate": pos_rate}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--max_windows", type=int, default=0,
                    help="0 = use all val windows")
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
    print(f"[model] encoder loaded  K={K} d_model={d_model} d_static={d_static}")

    if args.max_windows > 0 and args.max_windows < len(ds_val):
        # Deterministic head-subset to keep things fast for an exploratory probe.
        ds_val.windows = ds_val.windows[: args.max_windows]
        print(f"[data] truncated to first {len(ds_val)} windows")

    loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    t0 = time.time()
    feats = gather_features(enc, loader, device)
    n_videos = len({int(v.item()) for v in feats["video_id"]})
    print(f"[encode] {n_videos} val videos / {feats['latent_global'].shape[0]} windows "
          f"in {time.time()-t0:.1f}s")

    tr_set, te_set = split_by_video(feats["video_id"], args.probe_split_seed)
    print(f"[probe] {len(tr_set)} probe-train videos / {len(te_set)} probe-test videos "
          f"(seed={args.probe_split_seed})")

    results = {}

    # (A) raw Wan latent — multi-label "any visible slot has class c"
    results["A_raw_latent_global_multilabel"] = run_multilabel(
        "RAW WAN LATENT (global mean, 48-d)",
        feats["latent_global"], feats["attrs"], feats["slot_mask"], feats["video_id"],
        tr_set, te_set, device, args.probe_epochs, args.probe_lr, args.probe_wd,
    )

    # ----- (A2) Window-MAJORITY probes — escapes the multi-label imbalance trap -----
    # Build per-window majority labels (color: 8-way; material: 2-way; shape: 3-way)
    print("\n=== building per-window MAJORITY-class labels ===")
    n_win = feats["latent_global"].shape[0]
    flat_latent_path = args.cache_dir
    # full-flat raw latent: re-read each window's latent flattened to 9216-d
    # (we didn't keep this in `feats` to save memory; re-encode by iterating loader once more)
    flat_latent = []
    for batch in DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate):
        flat_latent.append(batch["latent"].reshape(batch["latent"].shape[0], -1).float())
    flat_latent = torch.cat(flat_latent, dim=0)        # (N, 48*T_lat*8*8) = (N, 9216)
    print(f"[probe] flat latent shape: {tuple(flat_latent.shape)}")

    maj_results = {}
    for gname, lo, hi, vocab in ATTR_GROUPS:
        nc = hi - lo
        # majority class per window
        N, K = feats["slot_mask"].shape
        y = torch.full((N,), -1, dtype=torch.long)
        for n in range(N):
            counts = torch.zeros(nc)
            for k in range(K):
                if not bool(feats["slot_mask"][n, k].item()):
                    continue
                oh = feats["attrs"][n, k, lo:hi]
                if oh.sum().item() < 0.5:
                    continue
                counts[int(oh.argmax().item())] += 1
            if counts.sum() > 0:
                y[n] = int(counts.argmax().item())
        keep = y >= 0
        y = y[keep]
        # split
        vids_arr = feats["video_id"][keep]
        keep_tr = torch.tensor([int(v.item()) in tr_set for v in vids_arr], dtype=torch.bool)
        keep_te = torch.tensor([int(v.item()) in te_set for v in vids_arr], dtype=torch.bool)

        gp = feats["latent_global"][keep]
        st = feats["static_192"][keep]         # need to aggregate over slots
        fl = flat_latent[keep]
        # per-window static_192 representation: mean over K slots (with slot_mask weight)
        sm = feats["slot_mask"][keep].float()
        st_mean = (st * sm.unsqueeze(-1)).sum(dim=1) / sm.sum(dim=1, keepdim=True).clamp_min(1)

        y_tr, y_te = y[keep_tr], y[keep_te]
        maj_test = float((torch.bincount(y_te, minlength=nc).max().item() /
                          max(y_te.numel(), 1)))

        print(f"\n=== MAJORITY [{gname}] — {nc} classes, n_train={keep_tr.sum().item()}, n_test={keep_te.sum().item()}, majority_baseline={maj_test:.3f} ===")
        row = {}
        for fname, F_ in [("global_pool_48d", gp),
                          ("static_192_winmean", st_mean),
                          ("flat_latent_9216d", fl)]:
            best = train_linear(F_[keep_tr], y_tr, F_[keep_te], y_te, nc,
                                epochs=args.probe_epochs, lr=args.probe_lr,
                                wd=max(args.probe_wd, 1e-3 if "flat" in fname else args.probe_wd),
                                device=device)
            print(f"   {fname:25s}: train {best['train_acc']:.3f}  TEST {best['test_acc']:.3f}  Δ vs maj {best['test_acc']-maj_test:+.3f}")
            row[fname] = {"train_acc": best["train_acc"], "test_acc": best["test_acc"]}
        maj_results[gname] = {"n_train": int(keep_tr.sum().item()),
                              "n_test": int(keep_te.sum().item()),
                              "num_classes": nc, "majority_baseline": maj_test,
                              "by_feature": row}
    results["A2_majority_class_per_window"] = maj_results

    # (B1) encoder pre-projection static_out (192-d) — per-slot multi-class
    results["B1_static_out_192_perslot"] = run_per_slot(
        "ENCODER static_out PRE-PROJECTION (192-d, per slot)",
        feats["static_192"], feats["attrs"], feats["slot_mask"], feats["video_id"],
        tr_set, te_set, device, args.probe_epochs, args.probe_lr, args.probe_wd,
    )

    # (B2) encoder pre-projection frame_slot mean (192-d) — per-slot multi-class
    results["B2_frame_slot_out_192_mean_perslot"] = run_per_slot(
        "ENCODER frame_slot_out PRE-PROJECTION (192-d mean, per slot)",
        feats["frame_192_mean"], feats["attrs"], feats["slot_mask"], feats["video_id"],
        tr_set, te_set, device, args.probe_epochs, args.probe_lr, args.probe_wd,
    )

    # ============ comparison summary ============
    print("\n=== summary: TEST accuracy across feature variants ===")
    print(f"{'group':10s} | "
          f"{'A raw (multilab)':>18s} | "
          f"{'B1 static_192':>14s} | "
          f"{'B2 frame_192':>13s} | "
          f"{'(ref: z_static_16 from prior probe)':>40s}")
    for g, _, _, _ in ATTR_GROUPS:
        a_acc = results["A_raw_latent_global_multilabel"][g]["test_acc"]
        a_base = results["A_raw_latent_global_multilabel"][g]["always_zero_baseline"]
        b1 = results["B1_static_out_192_perslot"][g]["test_acc"]
        b1m = results["B1_static_out_192_perslot"][g]["majority"]
        b2 = results["B2_frame_slot_out_192_mean_perslot"][g]["test_acc"]
        print(f"{g:10s} | {a_acc:>10.3f} (b0 {a_base:.3f}) | "
              f"{b1:>9.3f} (m {b1m:.3f}) | {b2:>8.3f}     |")

    out_path = Path(args.ckpt).parent / "probe_wan_latent.json"
    with open(out_path, "w") as f:
        json.dump({
            "protocol": "Experiment 1 — A: raw Wan latent global multi-label; "
                        "B1/B2: pre-projection encoder features per-slot multi-class.",
            "ckpt": args.ckpt,
            "cache_dir": args.cache_dir,
            "n_val_videos": n_videos,
            "n_windows": feats["latent_global"].shape[0],
            "probe_train_videos": len(tr_set),
            "probe_test_videos": len(te_set),
            "results": results,
        }, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
