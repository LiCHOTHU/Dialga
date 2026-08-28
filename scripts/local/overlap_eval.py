"""How much do z_static and z_dyn OVERLAP? (the redundancy term, measured directly)

Everything measured so far speaks to necessity (zs_cost/zd_cost: does removing a code
hurt?) or to separation of one factor (swap margin: does identity follow z_static?).
Neither is redundancy. And L_indep only rules out LINEAR correlation -- two codes can
be perfectly decorrelated (measured: 0.0037) and still encode the same content through
a nonlinear map.

Overlap is the PID "redundant" component: information present in BOTH codes. The
tractable proxy here is nonlinear predictability -- fit a small MLP from one code to
the other and report R^2. If z_dyn predicts z_static well, the pair is carrying the
same thing twice and the split is cosmetic, however necessary each code looks under
ablation.

  overlap_d2s  R^2 predicting z_static from z_dyn   (want LOW)
  overlap_s2d  R^2 predicting z_dyn    from z_static (want LOW)
  linear_xcorr the L_indep quantity, for contrast: it is what the loss optimises, and
               the point is that it can be ~0 while the nonlinear overlap is large.
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
from src.data.clevrer_sequence import ClevrerSequence          # noqa: E402
from scripts.local.eval_psnr import build                       # noqa: E402


def cka(X, Y):
    """Linear CKA between two representations. Needs no fitting, so unlike a learned
    probe it is reliable at ~1000 samples -- the MLP version of this test returned
    NEGATIVE R^2 (worse than predicting the mean), i.e. it overfit and measured
    nothing. 0 = unrelated, 1 = the same information up to linear transform."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro"); yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xty / (xx * yy + 1e-12))


def rbf_cka(X, Y, sigma_frac=0.5):
    """Kernel CKA -- catches NONLINEAR shared structure that linear CKA misses."""
    def K(A):
        d = ((A[:, None] - A[None]) ** 2).sum(-1)
        s = np.median(d[d > 0]) * sigma_frac + 1e-12
        Kk = np.exp(-d / (2 * s))
        n = len(A); H = np.eye(n) - np.ones((n, n)) / n
        return H @ Kk @ H
    Kx, Ky = K(X), K(Y)
    return float((Kx * Ky).sum() / (np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()) + 1e-12))


def mlp_r2(X, Y, dev, epochs=300, hid=512):
    """R^2 of a small MLP predicting Y from X, on a held-out half."""
    n = len(X) // 2
    Xtr = torch.tensor(X[:n], device=dev); Ytr = torch.tensor(Y[:n], device=dev)
    Xte = torch.tensor(X[n:], device=dev); Yte = torch.tensor(Y[n:], device=dev)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu, ysd = Ytr.mean(0, keepdim=True), Ytr.std(0, keepdim=True) + 1e-6
    Ytr_n, Yte_n = (Ytr - ymu) / ysd, (Yte - ymu) / ysd
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hid), torch.nn.SiLU(),
                              torch.nn.Linear(hid, hid), torch.nn.SiLU(),
                              torch.nn.Linear(hid, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.nn.functional.mse_loss(net(Xtr), Ytr_n)
        loss.backward(); opt.step()
    with torch.no_grad():
        p = net(Xte)
        ss_res = ((p - Yte_n) ** 2).sum()
        ss_tot = ((Yte_n - Yte_n.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


@torch.no_grad()
def codes(ck, loader, dev):
    enc, _, _ = build(ck, dev)
    S, D = [], []
    for b in loader:
        g, z, _ = enc(b["latents"].to(dev))
        S.append(g[:, 0].flatten(1).cpu()); D.append(z[:, 0].mean(1).cpu())
    return torch.cat(S).numpy(), torch.cat(D).numpy()


def lin_xcorr(A, B):
    a = (A - A.mean(0)) / (A.std(0) + 1e-4)
    b = (B - B.mean(0)) / (B.std(0) + 1e-4)
    return float((((a.T @ b) / (len(A) - 1)) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--dataset", choices=["clevrer", "ssv2"], default=None,
                    help="default: inferred from the cache path")
    ap.add_argument("--out", default="outputs/logs/overlap.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    # SSv2 caches carry no attrs/slot_mask, so the CLEVRER loader raises KeyError on
    # them. Pick the loader from the cache rather than assuming CLEVRER.
    ds = args.dataset or ("ssv2" if "ssv2" in Path(args.cache_dir).name.lower()
                          or "libero" in Path(args.cache_dir).name.lower() else "clevrer")
    if ds == "ssv2":
        from src.data.ssv2_sequence import SSv2Sequence
        va = SSv2Sequence(args.cache_dir, 4, 600, "val", preload=True)
    else:
        va = ClevrerSequence(args.cache_dir, 4, 600, "val", preload=True)
    dl = DataLoader(va, batch_size=32)
    print(f"[data] {len(va)} val videos\n")
    print(f"{'model':<22}{'CKA linear':>13}{'CKA rbf':>13}{'linear xcorr':>14}")
    print('-' * 63)
    res = {}
    for ck, lb in zip(args.ckpts, args.labels):
        S, D = codes(ck, dl, dev)
        idx = np.random.RandomState(0).permutation(len(S))[:800]
        r = {"cka_linear": cka(S[idx], D[idx]),
             "cka_rbf": rbf_cka(S[idx].astype(np.float64), D[idx].astype(np.float64)),
             "linear_xcorr": lin_xcorr(S, D)}
        res[lb] = r
        print(f"{lb:<22}{r['cka_linear']:>13.3f}{r['cka_rbf']:>13.3f}"
              f"{r['linear_xcorr']:>14.5f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\nCKA near 1 = the two codes carry the SAME information (redundant; the"
          "\nsplit is cosmetic). CKA near 0 = genuinely non-overlapping.")
    print("OVERLAP_DONE")


if __name__ == "__main__":
    main()
