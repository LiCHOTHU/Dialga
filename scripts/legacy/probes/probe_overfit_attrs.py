"""Overfit-accuracy probe for tiny 5-vid runs.

For overfit experiments (5 vids, 30 epochs), the meaningful question is
"does z_static now contain per-slot identity?" — measured as linear-readout
accuracy on the SAME 20 windows the encoder was trained on.

Reports:
  - linear-probe train accuracy on z_static (color, material, shape)
  - AttrsHead direct readout accuracy on the same data
  - SlotAttention final-attention entropy + max_prob (if encoder is SA)

Usage:
    python scripts/probe_overfit_attrs.py \\
        --cache_dir /storage/project/r-agarg35-0/lwang831/tmp/iter22_smoke_20260519_030044 \\
        --ckpt outputs/iter23_sa_5vid_<stamp>/trajectory.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.model.trajectory_encoder_v21 import (
    TrajectoryEncoder, TrajectoryEncoderSA, AttrsHead,
)
from scripts.legacy.train_trajectory import CachedLatentDataset, collate


GROUPS = [
    ("color",    0,                                       len(COLOR_VOCAB)),
    ("material", len(COLOR_VOCAB),                        len(COLOR_VOCAB) + len(MATERIAL_VOCAB)),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB),  len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB)),
]


@torch.no_grad()
def encode_all(enc, loader, device, capture_sa_attn=False):
    zs_chunks, at_chunks, sm_chunks = [], [], []
    sa_attn_entropy, sa_attn_maxprob = [], []
    for batch in loader:
        target = batch["latent"].to(device)
        if capture_sa_attn:
            # Manually unroll TrajectoryEncoderSA.forward to capture per-frame SlotAttention final-iter attention
            e = enc
            B, C, T, H, W = target.shape
            S = H * W
            x = target.permute(0, 2, 3, 4, 1).reshape(B, T, S, C)
            x = e.input_proj(x) + e.spatial_pos
            t_idx = torch.arange(T, device=x.device)
            x = x + e.temporal_pos(t_idx)[None, :, None, :]
            slots = e.slot_queries.expand(B, -1, -1).contiguous()
            for t in range(T):
                slots, attn = e.slot_attn(slots, x[:, t])     # attn: (B, K, S)
                ent = -(attn * attn.clamp_min(1e-12).log()).sum(-1)   # (B, K)
                mp = attn.max(-1).values                                # (B, K)
                sa_attn_entropy.append(ent.float().cpu().numpy())
                sa_attn_maxprob.append(mp.float().cpu().numpy())
            # Run rest of encoder via the public forward to get z_static.
        z_static, _, _, _ = enc(target)
        zs_chunks.append(z_static.float().cpu())
        at_chunks.append(batch["attrs"].clone())
        sm_chunks.append(batch["slot_mask"].clone().bool())
    return {
        "z_static": torch.cat(zs_chunks),
        "attrs":    torch.cat(at_chunks),
        "slot_mask":torch.cat(sm_chunks),
        "sa_attn_entropy": np.concatenate(sa_attn_entropy) if sa_attn_entropy else None,
        "sa_attn_maxprob": np.concatenate(sa_attn_maxprob) if sa_attn_maxprob else None,
    }


def per_slot_pairs(z_static, attrs, slot_mask, lo, hi):
    xs, ys = [], []
    N, K = slot_mask.shape
    for n in range(N):
        for k in range(K):
            if not bool(slot_mask[n, k].item()):
                continue
            oh = attrs[n, k, lo:hi]
            if oh.sum().item() < 0.5:
                continue
            xs.append(z_static[n, k])
            ys.append(int(oh.argmax().item()))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def train_linear(x, y, num_classes, epochs=2000, lr=1e-2, wd=1e-4, device="cuda"):
    probe = nn.Linear(x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    x, y = x.to(device), y.to(device)
    best_acc = 0.0
    for ep in range(epochs):
        probe.train()
        loss = F.cross_entropy(probe(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            probe.eval()
            with torch.no_grad():
                acc = (probe(x).argmax(-1) == y).float().mean().item()
            best_acc = max(best_acc, acc)
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = CachedLatentDataset(args.cache_dir, split="all", val_frac=0.0)
    print(f"[data] {len(ds)} windows (overfit mode, no train/val split)")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    use_sa = bool(a.get("use_slot_attn", False))
    K        = int(a.get("K", 8))
    d_model  = int(a.get("d_model", 192))
    n_heads  = int(a.get("n_heads", 4))
    n_layers = int(a.get("n_layers", 4))
    d_static = int(a.get("d_static", 16))
    d_dyn    = int(a.get("d_dyn", 32))

    sample = ds[0]
    C, T, H, W = sample["latent"].shape
    if use_sa:
        enc = TrajectoryEncoderSA(
            latent_ch=C, K=K, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            T_max=max(T * 2, 16), spatial_size=H, d_static=d_static, d_dyn=d_dyn,
            sa_iters=int(a.get("sa_iters", 3)), dropout=0.0,
        ).to(device)
    else:
        enc = TrajectoryEncoder(
            latent_ch=C, K=K, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            T_max=max(T * 2, 16), spatial_size=H, d_static=d_static, d_dyn=d_dyn,
            dropout=0.0,
        ).to(device)
    enc.load_state_dict(ckpt.get("encoder_state_dict", ckpt.get("encoder", ckpt)))
    enc.eval()
    print(f"[model] {'SA' if use_sa else 'baseline'} encoder loaded "
          f"({sum(p.numel() for p in enc.parameters())/1e6:.2f}M)")

    # AttrsHead reload (if present)
    attrs_head = None
    if "attrs_head_state_dict" in ckpt and ckpt["attrs_head_state_dict"] is not None:
        attrs_head = AttrsHead(d_static=d_static, hidden=int(a.get("attrs_hidden", 64))).to(device)
        attrs_head.load_state_dict(ckpt["attrs_head_state_dict"])
        attrs_head.eval()
        print("[model] AttrsHead loaded")

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    feats = encode_all(enc, loader, device, capture_sa_attn=use_sa)
    z_static = feats["z_static"]; attrs = feats["attrs"]; slot_mask = feats["slot_mask"]
    print(f"[encode] z_static {tuple(z_static.shape)}")

    print("\n=== Linear identity overfit accuracy on z_static ===")
    print(f"{'group':10s} {'classes':>8s} {'n_slots':>10s} {'chance':>8s} "
          f"{'majority':>10s} {'TRAIN_ACC':>10s} {'Δ vs maj':>10s}")
    results = {"linear_overfit": {}}
    for gname, lo, hi in GROUPS:
        x, y = per_slot_pairs(z_static, attrs, slot_mask, lo, hi)
        nc = hi - lo
        ch = 1.0 / nc
        maj = (torch.bincount(y, minlength=nc).max().item() / max(y.numel(), 1))
        acc = train_linear(x, y, nc, device=device)
        print(f"{gname:10s} {nc:>8d} {x.shape[0]:>10d} {ch:>8.3f} {maj:>10.3f} "
              f"{acc:>10.3f} {acc-maj:>+10.3f}")
        results["linear_overfit"][gname] = {
            "num_classes": nc, "n_slots": x.shape[0],
            "chance": ch, "majority": maj, "train_acc": acc,
        }

    if attrs_head is not None:
        print("\n=== AttrsHead direct readout on z_static ===")
        with torch.no_grad():
            lc, lm, ls = attrs_head(z_static.to(device))     # each (N, K, n_*)
        results["attrs_head_direct"] = {}
        N, Kdim = slot_mask.shape
        for gname, lo, hi, logits in [
            ("color",    0, 8,  lc),
            ("material", 8, 10, lm),
            ("shape",   10, 13, ls),
        ]:
            tgt = attrs[..., lo:hi].argmax(-1)
            mask_bool = slot_mask
            pred = logits.argmax(-1).cpu()
            correct = ((pred == tgt) & mask_bool).sum().item()
            total = mask_bool.sum().item()
            acc = correct / max(total, 1)
            nc = hi - lo
            tgt_flat = tgt[mask_bool]
            maj = (torch.bincount(tgt_flat, minlength=nc).max().item() / max(tgt_flat.numel(), 1))
            print(f"{gname:10s} {nc:>8d} {tgt_flat.numel():>10d} {1.0/nc:>8.3f} "
                  f"{maj:>10.3f} {acc:>10.3f} {acc-maj:>+10.3f}")
            results["attrs_head_direct"][gname] = {"acc": acc, "majority": maj}

    if feats["sa_attn_entropy"] is not None:
        ent_mean = float(feats["sa_attn_entropy"].mean())
        mp_mean = float(feats["sa_attn_maxprob"].mean())
        print(f"\n=== Slot-Attention attention sharpness (per-frame, final iter) ===")
        print(f"   entropy mean (lower = sharper, uniform = {np.log(64):.3f} nats): {ent_mean:.3f}")
        print(f"   max_prob mean (higher = sharper, uniform = {1/64:.4f}):           {mp_mean:.3f}")
        results["sa_attention_sharpness"] = {
            "entropy_mean_nats": ent_mean, "max_prob_mean": mp_mean,
            "uniform_entropy": float(np.log(64)), "uniform_max_prob": 1/64,
        }

    out_path = Path(args.ckpt).parent / "probe_overfit_attrs.json"
    with open(out_path, "w") as f:
        json.dump({"use_sa": use_sa, "ckpt": args.ckpt, "results": results}, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
