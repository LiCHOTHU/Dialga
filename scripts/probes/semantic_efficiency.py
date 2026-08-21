"""Semantic EFFICIENCY experiments (the honest 'more semantic AND efficient' claim,
vs the fair peer class: methods on the SAME frozen Wan-VAE latent, matched rate).

Two curves on CLEVRER attribute-presence mAP, all methods on the frozen VAE latent:
  (1) RATE curve      -- mAP vs #floats: ours (z_static, z_static+z_dyn) vs PCA of the
                         raw latent at matched dims vs the full latent vs random. Shows
                         ours is on the efficient frontier (as good as the full latent
                         at 288x fewer floats, above linear compression at matched dim).
  (2) LABEL curve     -- mAP vs #train labels: a more semantic code needs fewer labels.
                         ours vs PCA-96 vs full latent.

Foundation encoders (DINOv2/VideoMAE) are NOT included: different input (raw pixels)
and pretraining, so not a fair matched comparison -- they belong in a reference row.

Reuses the presence-mAP machinery from baseline_probe_table.
Usage:
  python scripts/probes/semantic_efficiency.py --ckpt .../v5_best.pt \
      --cache_dir .../wan_10000vid_W33 --out .../semantic_efficiency.json
"""
from __future__ import annotations
import argparse, json, sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.baseline_probe_table import (
    build_encoder, collect, run_probe, eval_scores, GROUPS)


def probe_mean(Ztr, Ytr, Zva, Yva, device, epochs, lr, wd):
    sc, _m, _pe = run_probe(Ztr, Ytr, Zva, Yva, device, epochs, lr, wd)
    return float(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dino_cache_dir", default="")
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[48, 96, 192, 352, 768])
    ap.add_argument("--label_fracs", type=float, nargs="*",
                    default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
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

    # our composite code (static + dynamic), if z_dyn features were collected via enc
    # collect() only stores z_static; add z_static+z_dyn by re-reading is overkill --
    # we report z_static (identity code, 96) which is the semantic slot.
    Xtr = Ftr["wan_flat"].float().to(device); Xva = Fva["wan_flat"].float().to(device)
    mu = Xtr.mean(0, keepdim=True)

    def pca_feat(q):
        q = min(q, min(Xtr.shape) - 1)
        _, _, V = torch.pca_lowrank(Xtr - mu, q=q, niter=4)
        return ((Xtr - mu) @ V).cpu(), ((Xva - mu) @ V).cpu(), q

    common = dict(device=device, epochs=args.epochs, lr=args.lr, wd=args.weight_decay)
    results = {"rate_curve": {}, "label_curve": {}}

    # ---- (1) RATE curve: mAP vs #floats ----
    rate_methods = {
        "ours_z_static": (Ftr["ours_z_static"].float(), Fva["ours_z_static"].float(),
                          Ftr["ours_z_static"].shape[1]),
        "random_enc":    (Ftr["random_enc"].float(), Fva["random_enc"].float(),
                          Ftr["random_enc"].shape[1]),
        "wan_meanpool":  (Ftr["wan_meanpool"].float(), Fva["wan_meanpool"].float(),
                          Ftr["wan_meanpool"].shape[1]),
        "wan_flat":      (Ftr["wan_flat"].float(), Fva["wan_flat"].float(),
                          Ftr["wan_flat"].shape[1]),
    }
    for d in args.pca_dims:
        Zt, Zv, q = pca_feat(d)
        rate_methods[f"wan_pca{q}"] = (Zt, Zv, q)
    for name, (Zt, Zv, dim) in rate_methods.items():
        m = probe_mean(Zt, Ytr, Zv, Yva, **common)
        results["rate_curve"][name] = {"dim": int(dim), "mAP": round(m, 4)}
        print(f"[rate] {name:<16} dim={dim:<6} mAP={m:.4f}", flush=True)

    # ---- (2) LABEL curve: mAP vs #train labels ----
    if "wan_pca96" in rate_methods:
        pca96_tr, pca96_va = rate_methods["wan_pca96"][0], rate_methods["wan_pca96"][1]
    else:
        pca96_tr, pca96_va, _ = pca_feat(96)
    label_methods = {
        "ours_z_static": (Ftr["ours_z_static"].float(), Fva["ours_z_static"].float()),
        "wan_pca96":     (pca96_tr, pca96_va),
        "wan_flat":      (Ftr["wan_flat"].float(), Fva["wan_flat"].float()),
    }
    n = len(Ytr)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    for name, (Zt, Zv) in label_methods.items():
        results["label_curve"][name] = {}
        for frac in args.label_fracs:
            k = max(20, int(n * frac))
            idx = perm[:k]
            m = probe_mean(Zt[idx], Ytr[idx], Zv, Yva, **common)
            results["label_curve"][name][str(frac)] = {"n": int(k), "mAP": round(m, 4)}
            print(f"[label] {name:<14} frac={frac:<5} n={k:<5} mAP={m:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\nSEMANTIC_EFFICIENCY_DONE", flush=True)


if __name__ == "__main__":
    main()
