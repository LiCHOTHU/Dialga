"""Stage-0 diagnosis for the MAETok plan: is identity LINEARLY absent or ABSENT?

Same protocol as probe_v5_modal.py (scene-level modal attribute, 50/50 video
split, seed 0) but run on the cached val embeddings from
cache_val_embeddings.py, with BOTH a linear probe and a 2-layer MLP probe
(hidden 256, ReLU, dropout 0.1) on the same split.

Decision rule (from the improvement plan):
  - MLP ~ linear (both weak)  -> info absent from z_static -> MAETok aux
    semantic loss is justified.
  - MLP >> linear (color > 0.7) -> info present nonlinearly -> the decoder,
    not the latent, is the bottleneck.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# CLEVRER attr one-hot layout (matches src/data/clevrer_states.py vocabs)
COLOR_N, MATERIAL_N, SHAPE_N = 8, 2, 3
ATTR_GROUPS = [
    ("color", 0, COLOR_N),
    ("material", COLOR_N, COLOR_N + MATERIAL_N),
    ("shape", COLOR_N + MATERIAL_N, COLOR_N + MATERIAL_N + SHAPE_N),
]


def modal_label(attrs: torch.Tensor, slot_mask: torch.Tensor, lo: int, hi: int) -> int:
    real_k = slot_mask.bool()
    if not real_k.any():
        return -1
    classes = attrs[real_k, lo:hi].argmax(dim=-1)
    counts = torch.bincount(classes, minlength=hi - lo)
    return int(counts.argmax().item())


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_probe(probe, x_tr, y_tr, x_te, y_te, epochs, lr, weight_decay, device):
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    best = {"train_acc": 0.0, "test_acc": 0.0, "ep": 0, "loss": float("inf")}
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
                best = {"train_acc": tr_acc, "test_acc": te_acc, "ep": ep, "loss": loss.item()}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="/storage/scratch1/8/lwang831/dialga_outputs/"
                    "v512_e1_frozen_20260526_001020/val_embeddings.pt")
    ap.add_argument("--feature", default="video_z_static",
                    choices=["video_z_static", "video_z_dyn_mean", "video_wan_mean"])
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--probe_epochs", type=int, default=2000)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    feats = data[args.feature]                       # (N_vid, D)
    video_ids = [int(v) for v in data["video_ids"]]
    per_video = data["per_video"]
    print(f"[data] {args.feature} {tuple(feats.shape)} from {args.embeddings}")

    # 50/50 split by video id, same seed as probe_v5_modal.py
    order = sorted(range(len(video_ids)), key=lambda i: video_ids[i])
    rng = random.Random(args.probe_split_seed)
    rng.shuffle(order)
    half = len(order) // 2
    tr_idx, te_idx = order[:half], order[half:]

    def build_xy(idxs, lo, hi):
        xs, ys = [], []
        for i in idxs:
            vid = video_ids[i]
            rec = per_video[vid] if vid in per_video else per_video[str(vid)]
            y = modal_label(rec["attrs"], rec["slot_mask"], lo, hi)
            if y < 0:
                continue
            xs.append(feats[i])
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    results = {}
    hdr = (f"{'group':10s} {'maj':>7s} | {'lin tr':>7s} {'lin te':>7s} {'Δlin':>7s} | "
           f"{'mlp tr':>7s} {'mlp te':>7s} {'Δmlp':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, lo, hi in ATTR_GROUPS:
        x_tr, y_tr = build_xy(tr_idx, lo, hi)
        x_te, y_te = build_xy(te_idx, lo, hi)
        num_classes = hi - lo
        majority = (torch.bincount(y_te, minlength=num_classes).max().item()
                    / max(y_te.numel(), 1))
        torch.manual_seed(0)
        lin = train_probe(nn.Linear(x_tr.shape[1], num_classes),
                          x_tr, y_tr, x_te, y_te,
                          args.probe_epochs, args.probe_lr, args.probe_wd, args.device)
        torch.manual_seed(0)
        mlp = train_probe(MLPProbe(x_tr.shape[1], num_classes, args.mlp_hidden),
                          x_tr, y_tr, x_te, y_te,
                          args.probe_epochs, args.probe_lr, args.probe_wd, args.device)
        print(f"{name:10s} {majority:>7.3f} | "
              f"{lin['train_acc']:>7.3f} {lin['test_acc']:>7.3f} {lin['test_acc']-majority:>+7.3f} | "
              f"{mlp['train_acc']:>7.3f} {mlp['test_acc']:>7.3f} {mlp['test_acc']-majority:>+7.3f}")
        results[name] = {
            "majority": majority,
            "linear": lin,
            "mlp": mlp,
        }

    out_path = Path(args.embeddings).parent / f"probe_v5_nonlinear_{args.feature}.json"
    with open(out_path, "w") as f:
        json.dump({"feature": args.feature, "embeddings": args.embeddings,
                   "mlp_hidden": args.mlp_hidden, "results": results}, f, indent=2)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
