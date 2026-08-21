"""Decision gate for the simple-transformer + MAE test ([[reference-audit-2026-06-28]]).

Trains VitMAE on a CLEVRER subset (Wan-latent space, NO DINO) until the masked
reconstruction loss plateaus, then answers "is the representation useful?":

  1. recon: train + val MASKED recon loss (does MAE learn structure / generalize).
  2. usefulness: a fresh LINEAR probe on the FROZEN MAE representation (mean-pooled
     encoder tokens), trained on the train split, evaluated on val -> color/material/
     shape top-1 accuracy + majority baseline. Directly comparable to the mean-pool
     baseline (val attrs ~62.9% from eval_attrs_accuracy.py).
  3. efficiency: participation ratio of the pooled representation on val.

Standalone, local GPU. Example:
  python scripts/probes/vit_mae_gate.py --cache_dir .../wan_10000vid_W33 \
      --max_videos 500 --epochs 60 --mask_ratio 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.vit_mae import VitMAE

N_COLOR, N_MATERIAL, N_SHAPE = len(COLOR_VOCAB), len(MATERIAL_VOCAB), len(SHAPE_VOCAB)


def modal_labels(attrs, slot_mask, lo, hi):
    block = attrs[..., lo:hi]
    cls = block.argmax(dim=-1)
    onehot = F.one_hot(cls, hi - lo).float()
    masked = onehot * slot_mask.float().unsqueeze(-1)
    counts = masked.sum(dim=1)
    labels = counts.argmax(dim=-1)
    valid = slot_mask.bool().any(dim=-1)
    return torch.where(valid, labels, torch.full_like(labels, -100))


def labels_for(batch, device):
    a = batch["attrs"].to(device); sm = batch["slot_mask"].to(device)
    return {"color": modal_labels(a, sm, 0, N_COLOR),
            "material": modal_labels(a, sm, N_COLOR, N_COLOR + N_MATERIAL),
            "shape": modal_labels(a, sm, N_COLOR + N_MATERIAL,
                                  N_COLOR + N_MATERIAL + N_SHAPE)}


def participation_ratio(Z):
    Zc = Z - Z.mean(0, keepdim=True)
    cov = (Zc.T @ Zc) / max(Z.shape[0] - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    pr = (eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12)
    return float(pr), Z.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.6)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tr = ClevrerChunkPairs(args.cache_dir, split="train", val_frac=args.val_frac,
                           seed=args.seed, max_videos=args.max_videos)
    va = ClevrerChunkPairs(args.cache_dir, split="val", val_frac=args.val_frac,
                           seed=args.seed, max_videos=args.max_videos)
    trdl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                      collate_fn=chunk_collate, drop_last=True)
    vadl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4,
                      collate_fn=chunk_collate)
    print(f"[data] train pairs={len(tr)} val pairs={len(va)}")

    model = VitMAE(dim=args.dim, depth=args.depth, mask_ratio=args.mask_ratio).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] VitMAE dim={args.dim} depth={args.depth} mask={args.mask_ratio} "
          f"params={n_params/1e6:.2f}M tokens={model.n_tokens}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # ---- train MAE ----
    print("\n[train] epoch  train_masked  val_masked")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = n = 0.0
        for batch in trdl:
            x = batch["chunk_obs"].to(dev)
            out = model(x)
            opt.zero_grad(); out["loss"].backward(); opt.step()
            tl += out["loss"].item() * x.shape[0]; n += x.shape[0]
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            model.eval(); vl = vn = 0.0
            with torch.no_grad():
                for batch in vadl:
                    x = batch["chunk_obs"].to(dev)
                    o = model(x); vl += o["loss"].item() * x.shape[0]; vn += x.shape[0]
            print(f"        {ep:5d}  {tl/n:11.5f}  {vl/vn:10.5f}")

    # ---- representation: mean-pooled encoder tokens (frozen) ----
    model.eval()
    def reps(dl):
        Z, Y = [], {"color": [], "material": [], "shape": []}
        with torch.no_grad():
            for batch in dl:
                tok = model.encode(batch["chunk_obs"].to(dev))   # (B, N, dim)
                Z.append(tok.mean(dim=1).cpu())                  # mean-pool readout
                ys = labels_for(batch, dev)
                for k in Y: Y[k].append(ys[k].cpu())
        return torch.cat(Z), {k: torch.cat(v) for k, v in Y.items()}
    Ztr, Ytr = reps(trdl); Zva, Yva = reps(vadl)

    # ---- linear probe (fresh, trained on train reps, eval on val) ----
    print("\n[probe] linear attrs probe on frozen MAE representation")
    probes = {"color": N_COLOR, "material": N_MATERIAL, "shape": N_SHAPE}
    accs, bases = {}, {}
    for k, ncls in probes.items():
        clf = nn.Linear(Ztr.shape[1], ncls).to(dev)
        po = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
        ztr, ytr = Ztr.to(dev), Ytr[k].to(dev)
        m = ytr != -100
        for _ in range(args.probe_epochs):
            po.zero_grad()
            loss = F.cross_entropy(clf(ztr[m]), ytr[m])
            loss.backward(); po.step()
        with torch.no_grad():
            zva, yva = Zva.to(dev), Yva[k].to(dev)
            mv = yva != -100
            pred = clf(zva[mv]).argmax(-1)
            accs[k] = float((pred == yva[mv]).float().mean())
            bases[k] = float(torch.bincount(yva[mv]).max()) / int(mv.sum())
    mean_acc = sum(accs.values()) / 3
    pr, D = participation_ratio(Zva)

    print(f"\n{'='*66}")
    print("RESULT (simple transformer + MAE, Wan-latent target)")
    print(f"{'='*66}")
    print(f"  val attrs:  color {accs['color']*100:.1f}% (maj {bases['color']*100:.1f}%)  "
          f"matl {accs['material']*100:.1f}% (maj {bases['material']*100:.1f}%)  "
          f"shape {accs['shape']*100:.1f}% (maj {bases['shape']*100:.1f}%)")
    print(f"  val attrs MEAN: {mean_acc*100:.1f}%   [mean-pool baseline ~62.9%]")
    print(f"  participation ratio: {pr:.2f} / {D}  ({pr/D*100:.1f}%)")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
