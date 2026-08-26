"""How many floats does the TIME-CONSTANT part of a chunk actually need?

Zero training. Under the base+delta decoder z_static alone must produce mean_t(x),
which carries ~54% of a chunk's latent energy, through 96 floats on a 4x4 grid --
where the old decoder let z_dyn's 2304 floats cover it. If reconstruction got worse,
the first question is whether 96 floats can represent that target AT ALL.

PCA on mean_t(x) (48x8x8 = 3072 dims) gives the information-theoretic best linear
code at each budget, so it upper-bounds what any encoder could do at that rate. If
PCA-96 is already lossy, the fix is rate, not architecture -- and the rate is worth
paying only now that the decoder actually depends on z_static.

Also reports the 4x4-grid spatial bottleneck separately: the current z_static is not
just 96 numbers, it is 6 channels on a 4x4 lattice bilinearly upsampled to 8x8, which
is a second, independent loss on top of the rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--n", type=int, default=4000, help="chunks to fit PCA on")
    ap.add_argument("--out", default="outputs/logs/diag_static_rate.json")
    args = ap.parse_args()

    files = sorted((Path(args.cache_dir) / "latents").glob("*.pt"))[: args.n]
    M, X = [], []
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=False)["latent"].float()
        M.append(x.mean(dim=1))                      # (C,H,W) the static target
        X.append(x)
    M = torch.stack(M)                               # (N,C,H,W)
    N, C, H, W = M.shape
    flat = M.reshape(N, -1).cuda()                   # (N, 3072)
    energy_full = sum(float((x ** 2).sum()) for x in X)
    energy_static = float((M ** 2).sum())
    print(f"[data] {N} chunks | static target {C}x{H}x{W} = {C*H*W} floats")
    print(f"[data] time-constant share of chunk energy: "
          f"{energy_static*len(X[0][0][0])/max(1e-9,energy_full)*0+energy_static/energy_full*X[0].shape[1]:.1%}")

    mu = flat.mean(0, keepdim=True)
    Z = flat - mu
    # economy SVD -> principal directions of the static target
    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    tot = float((Z ** 2).sum())
    res = {"n_chunks": N, "target_dims": C * H * W, "pca": {}, "grid": {}}

    print("\n=== PCA rate curve on mean_t(x): best possible linear code at each budget ===")
    print(f"{'floats':>8}{'rel-MSE':>10}{'var kept':>10}")
    for k in (48, 96, 192, 256, 384, 768, 1536):
        if k > min(N, C * H * W):
            continue
        kept = float((S[:k] ** 2).sum())
        rel = 1.0 - kept / tot
        res["pca"][k] = rel
        print(f"{k:>8}{rel:>10.4f}{kept/tot:>10.1%}")

    print("\n=== spatial bottleneck: 4x4 (current) vs 8x8 static grid, rate held ===")
    for g in (4, 6, 8):
        down = F.adaptive_avg_pool2d(M.cuda(), (g, g))
        up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
        rel = float(((up - M.cuda()) ** 2).sum() / (M.cuda() ** 2).sum())
        res["grid"][g] = rel
        print(f"  grid {g}x{g}: rel-MSE {rel:.4f}   (pure resolution loss, all 48 channels kept)")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n[saved] {args.out}")
    print("RATE_DONE")


if __name__ == "__main__":
    main()
