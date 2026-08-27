"""Raw-latent SSv2 ceiling, measured on the SAME cache and split as our model.

DEVLOG records the full raw Wan latent probing SSv2 at 9.5% top-1, described as "the
ceiling of what any encoder could extract from this VAE". Our 896-float z_dyn reads
10.43% on the full cache -- ABOVE it. Either a real result (a compressed code beating
the latent it is distilled from, which the label-efficiency story predicts) or evidence
that the 9.5% was underfit: a linear probe on 27,648 dims needs a great deal of data,
and that number came from an 8k-clip subset.

This re-measures it on the 163,717/28,891 split the model actually trained on, so the
comparison is like-for-like. Needs no checkpoint -- it probes the raw latent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.ssv2_sequence import SSv2Sequence                  # noqa: E402


def probe(Xtr, ytr, Xte, yte, name):
    """Linear probe on GPU (sklearn is impractical at 160k x 15k)."""
    dev = "cuda"
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    n_cls = int(max(ytr.max(), yte.max())) + 1
    W = torch.zeros(Xtr.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=1e-3, weight_decay=1e-4)
    N = len(Xtr)
    for ep in range(40):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, 4096):
            idx = perm[i:i + 4096]
            loss = torch.nn.functional.cross_entropy(Xtr[idx] @ W + b, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = ((Xte @ W + b).argmax(1) == yte).float().mean().item()
    print(f"  {name:<28} dim {Xtr.shape[1]:>6}   top-1 {acc*100:6.2f}%", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="outputs/cache/ssv2_W17_full")
    ap.add_argument("--max_train", type=int, default=40000)
    ap.add_argument("--out", default="outputs/logs/ssv2_ceiling.json")
    args = ap.parse_args()
    dev = "cuda"

    tr = SSv2Sequence(args.cache_dir, 4, args.max_train, "train")
    va = SSv2Sequence(args.cache_dir, 4, 0, "val")
    print(f"[data] {len(tr)} train / {len(va)} val clips", flush=True)

    def gather(ds):
        flat, mean, lab = [], [], []
        for b in DataLoader(ds, batch_size=64, num_workers=6):
            x = b["latents"][:, 0]                       # (B,C,T,H,W) first chunk
            flat.append(x.flatten(1))
            mean.append(x.mean(dim=(2, 3, 4)))
            lab.append(b["label_id"])
        return (torch.cat(flat), torch.cat(mean), torch.cat(lab))

    Ftr, Mtr, ytr = gather(tr)
    Fte, Mte, yte = gather(va)
    keep = (ytr >= 0)
    Ftr, Mtr, ytr = Ftr[keep], Mtr[keep], ytr[keep]
    keep = (yte >= 0)
    Fte, Mte, yte = Fte[keep], Mte[keep], yte[keep]
    print(f"[data] labelled: {len(ytr)} train / {len(yte)} val, "
          f"{int(yte.max())+1} classes, chance {1/ (int(yte.max())+1) * 100:.2f}%\n")

    res = {}
    res["wanmean_48"] = probe(Mtr.to(dev), ytr.to(dev), Mte.to(dev), yte.to(dev),
                              "raw Wan latent, mean-pool")
    res["wanflat_full"] = probe(Ftr.to(dev), ytr.to(dev), Fte.to(dev), yte.to(dev),
                                "raw Wan latent, FULL (ceiling)")
    print(f"\nour committed model on this split: z_dyn 10.43%, z_static 6.41%")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("CEILING_OK")


if __name__ == "__main__":
    main()
