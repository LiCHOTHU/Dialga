"""Baseline table for the video-EMBEDDING claim. This is the paper's gate.

WHY THIS IS THE WHOLE PAPER
---------------------------
`frozen_set_probe.py` showed our z_static hits colour mAP 0.932 against a 0.498
base-rate prior. That only proves z_static beats NOTHING. It does not prove it
beats the alternatives a reviewer will name in the first paragraph of their
review, and every one of those alternatives is free:

  ours_z_static   96      our learned embedding
  wan_meanpool    48      mean of the frozen Wan latent over (t,h,w). The dumbest
                          possible baseline. If this ties us, we have no paper.
  wan_pca96       96      PCA of the flat Wan latent, DIM-MATCHED to ours. The
                          critical control: does *learning* beat a linear
                          projection at identical width?
  wan_flat        27648   the entire Wan latent (48*9*8*8). Upper bound — the
                          most any method could extract from this VAE. Our claim
                          is 96 floats retaining what 27648 hold = 288x smaller.
  random_enc      96      our architecture, UNTRAINED. Separates "the encoder
                          learned something" from "conv features are just good".
  dino_meanpool   768     DINOv2 patch features, mean-pooled. Strong SSL baseline
                          and the target our MAE loss distils from.

The claim only survives if:  ours > wan_pca96, ours > random_enc, ours ~ wan_flat.
If wan_meanpool or random_enc ties us, the result is that our training is inert,
and that is the finding — we report it rather than bury it.

Same protocol as frozen_set_probe: FROZEN features -> single Linear -> BCE over
the permutation-invariant multi-hot presence set, scored as val mAP with
best-val selection. Features standardised on TRAIN stats only.
"""
from __future__ import annotations
import argparse, json, sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.latent_encoder import LatentEncoder3D

N_COLOR, N_MATERIAL, N_SHAPE = len(COLOR_VOCAB), len(MATERIAL_VOCAB), len(SHAPE_VOCAB)
GROUPS = [("color", 0, N_COLOR),
          ("material", N_COLOR, N_COLOR + N_MATERIAL),
          ("shape", N_COLOR + N_MATERIAL, N_COLOR + N_MATERIAL + N_SHAPE)]
N_ALL = N_COLOR + N_MATERIAL + N_SHAPE


def presence_labels(attrs, slot_mask):
    m = slot_mask.float().unsqueeze(-1)
    return (attrs.float() * m).sum(dim=1).clamp(max=1.0)


def average_precision(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / lab.sum())


def eval_scores(logits, Y):
    s, y = logits.numpy(), Y.numpy()
    out = {}
    for name, lo, hi in GROUPS:
        aps = [average_precision(s[:, c], y[:, c]) for c in range(lo, hi)]
        aps = [x for x in aps if not np.isnan(x)]
        out[name] = float(np.mean(aps)) if aps else float("nan")
    return out


def build_encoder(a, state, device, random_init=False):
    enc = LatentEncoder3D(
        d_static=a.d_static, d_dyn=a.d_dyn, hidden_ch=a.enc_hidden_ch,
        use_layer_norm=("norm_static.weight" in state),
        shared_trunk=getattr(a, "shared_trunk", False),
        pool_type=getattr(a, "pool_type", "mean"),
        static_grid=int(getattr(a, "static_grid", 4) or 4),
        n_queries=getattr(a, "pool_queries", 8),
        n_heads=getattr(a, "pool_heads", 4),
        chunk_size_lat=int(getattr(a, "chunk_size_lat", 9)),
        static_agg=getattr(a, "static_agg", "conv"),
        dyn_spatial=bool(getattr(a, "dyn_spatial", False)),
        dyn_grid=int(getattr(a, "dyn_grid", 8)),
    ).to(device)
    if not random_init:
        enc.load_state_dict(state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def collect(split, a, enc, rnd, device, dino_cache, max_batches):
    """One pass over the split -> every featurizer at once."""
    ds = ClevrerChunkPairs(a.cache_dir, split=split, val_frac=a.val_frac,
                           seed=a.seed, max_videos=a.max_videos,
                           dino_cache_dir=dino_cache)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=chunk_collate)
    feats = {k: [] for k in ("ours_z_static", "wan_meanpool", "wan_flat",
                             "random_enc", "dino_meanpool")}
    Y = []
    for bi, b in enumerate(dl):
        if max_batches and bi >= max_batches:
            break
        x = b["chunk_obs"].to(device)                       # (B,48,9,8,8)
        feats["ours_z_static"].append(enc(x)["z_static"].float().cpu())
        feats["random_enc"].append(rnd(x)["z_static"].float().cpu())
        feats["wan_meanpool"].append(x.mean(dim=(2, 3, 4)).float().cpu())
        feats["wan_flat"].append(x.flatten(1).half().cpu())
        if "dino_obs" in b:
            d = b["dino_obs"].to(device).float()            # (B,T,H,W,D)
            feats["dino_meanpool"].append(d.mean(dim=(1, 2, 3)).cpu())
        Y.append(presence_labels(b["attrs"], b["slot_mask"]))
    out = {k: torch.cat(v) for k, v in feats.items() if v}
    return out, torch.cat(Y)


def run_probe(Ztr, Ytr, Zva, Yva, device, epochs, lr, wd):
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True) + 1e-6
    Ztr, Zva = (Ztr - mu) / sd, (Zva - mu) / sd
    probe = nn.Linear(Ztr.shape[1], N_ALL).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Ztr_d, Ytr_d, Zva_d = Ztr.to(device), Ytr.to(device), Zva.to(device)
    best = None
    for ep in range(epochs):
        probe.train(); opt.zero_grad()
        lossf(probe(Ztr_d), Ytr_d).backward(); opt.step()
        if (ep + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                m = eval_scores(probe(Zva_d).cpu(), Yva)
            sc = np.nanmean([m[g] for g, _, _ in GROUPS])
            if best is None or sc > best[0]:
                best = (sc, m, ep + 1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="our trained checkpoint")
    ap.add_argument("--dino_cache_dir", default="")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pca_dim", type=int, default=96)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = Namespace(**a) if isinstance(a, dict) else a
    enc = build_encoder(a, ck["encoder"], device)
    rnd = build_encoder(a, ck["encoder"], device, random_init=True)

    dino = args.dino_cache_dir or None
    Ftr, Ytr = collect("train", a, enc, rnd, device, dino, args.max_batches)
    Fva, Yva = collect("val", a, enc, rnd, device, dino, args.max_batches)
    print(f"[data] train={len(Ytr)} val={len(Yva)}", flush=True)

    # dim-matched unsupervised control: PCA of the flat Wan latent, fit on TRAIN
    Xtr = Ftr["wan_flat"].float().to(device)
    Xva = Fva["wan_flat"].float().to(device)
    mu = Xtr.mean(0, keepdim=True)
    q = min(args.pca_dim, min(Xtr.shape) - 1)
    _, _, V = torch.pca_lowrank(Xtr - mu, q=q, niter=4)
    Ftr[f"wan_pca{q}"] = ((Xtr - mu) @ V).cpu()
    Fva[f"wan_pca{q}"] = ((Xva - mu) @ V).cpu()
    del Xtr, Xva
    torch.cuda.empty_cache()
    print(f"[pca] fit {q} components on train wan_flat", flush=True)

    prior = eval_scores(Ytr.mean(dim=0, keepdim=True).expand_as(Yva), Yva)
    rows = []
    for name in ("ours_z_static", f"wan_pca{q}", "wan_meanpool", "wan_flat",
                 "random_enc", "dino_meanpool"):
        if name not in Ftr:
            continue
        try:
            sc, m, pe = run_probe(Ftr[name].float(), Ytr, Fva[name].float(), Yva,
                                  device, args.epochs, args.lr, args.weight_decay)
            rows.append((name, Ftr[name].shape[1], m, sc, pe))
            print(f"  {name:<16} d={Ftr[name].shape[1]:<6} color={m['color']:.3f} "
                  f"material={m['material']:.3f} shape={m['shape']:.3f} mean={sc:.3f}",
                  flush=True)
        except Exception as e:
            print(f"  {name:<16} FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'='*80}\nFrozen linear probe, val mAP (higher=better)\n{'='*80}")
    print(f"{'features':<18}{'dim':>7}{'color':>9}{'material':>10}{'shape':>8}{'mean':>8}")
    print("-" * 80)
    pm = np.nanmean([prior[g] for g, _, _ in GROUPS])
    print(f"{'base-rate prior':<18}{'':>7}{prior['color']:>9.3f}{prior['material']:>10.3f}"
          f"{prior['shape']:>8.3f}{pm:>8.3f}")
    for name, d, m, sc, _ in sorted(rows, key=lambda r: -r[3]):
        print(f"{name:<18}{d:>7}{m['color']:>9.3f}{m['material']:>10.3f}"
              f"{m['shape']:>8.3f}{sc:>8.3f}")
    print("\nClaim survives only if ours_z_static > wan_pca* and > random_enc, "
          "and is close to wan_flat at 288x fewer dims.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"prior": prior, "ckpt": args.ckpt,
                   "rows": [{"features": n, "dim": d, "mAP": m, "mean": s, "probe_ep": pe}
                            for n, d, m, s, pe in rows]},
                  open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")
    print("BASELINE_PROBE_DONE")


if __name__ == "__main__":
    main()
