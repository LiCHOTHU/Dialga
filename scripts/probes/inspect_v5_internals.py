"""Model-internals inspection for a v5.1.x checkpoint.

Answers, with numbers, four questions about WHY the model under-performs:

  A. Activation health  - are encoder/decoder hidden channels alive?
                          (per-SiLU-layer: inert-channel fraction, mean |act|)
  B. Latent geometry    - how is information distributed in z?
                          per-dim std, participation ratio (effective dims),
                          pairwise-cosine collapse metric, train vs val.
  C. Decoder usage      - which z dims does the decoder actually listen to?
                          (per-dim output sensitivity to +1sigma perturbation)
  D. Pooling bottleneck - how much pre-pool spatial signal does the global
                          mean-pool destroy? (cell-norm spread vs pooled norm,
                          object-cell vs background-cell feature distance)

Latent-only (no VAE) so it runs on the local V100 in minutes.

Usage:
    python scripts/probes/inspect_v5_internals.py \
        --ckpt .../last.pt --cache_dir .../wan_10000vid_W33 \
        [--val_frac 0.2] [--n_videos 200]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_window import ClevrerChunkPairs
from src.model.latent_decoder import LatentDecoder, SpatialBroadcastDecoder
from src.model.latent_encoder import LatentEncoder3D


def participation_ratio(X):
    """X (N, D). PR = (sum eig)^2 / sum eig^2 of covariance — effective #dims."""
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0, None)
    s = eig.sum()
    return float(s * s / max((eig * eig).sum(), 1e-12)), eig[::-1]


def pairwise_cos(X):
    """Mean pairwise cosine of raw vectors. ->1.0 means collapsed."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    G = Xn @ Xn.T
    n = len(X)
    return float((G.sum() - n) / max(n * (n - 1), 1))


def collect_z(enc, ds, idxs, device, hooks_store=None):
    zs, zd, chunks = [], [], []
    with torch.no_grad():
        for i0 in range(0, len(idxs), 32):
            batch = torch.stack([ds[i]["chunk_obs"] for i in idxs[i0:i0 + 32]]).to(device)
            o = enc(batch)
            zs.append(o["z_static"].cpu())
            zd.append(o["z_dyn"].cpu())
            if hooks_store is not None and i0 == 0:
                chunks.append(batch.cpu())
    return torch.cat(zs).numpy(), torch.cat(zd).numpy()


def one_idx_per_video(ds, limit):
    seen, out = set(), []
    for i in range(len(ds)):
        vid = int(ds.windows[ds.pairs[i][0]]["video_id"])
        if vid in seen:
            continue
        seen.add(vid)
        out.append(i)
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--n_videos", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    use_ln = "norm_static.weight" in ck["encoder"]
    pool_type = a.get("pool_type", "mean")
    decoder_type = a.get("decoder_type", "linear")
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=pool_type,
                          n_queries=int(a.get("pool_queries", 8)),
                          n_heads=int(a.get("pool_heads", 4))).to(device)
    if decoder_type == "broadcast":
        dec = SpatialBroadcastDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                                      hidden_ch=int(a["dec_hidden_ch"]),
                                      chunk_size_lat=int(a.get("chunk_size_lat", 9)),
                                      depth=int(a.get("dec_depth", 3))).to(device)
    else:
        dec = LatentDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                            hidden_ch=int(a["dec_hidden_ch"]),
                            chunk_size_lat=int(a.get("chunk_size_lat", 9))).to(device)
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
    enc.eval(); dec.eval()
    print(f"[ckpt] ep={ck.get('epoch')} d_static={a['d_static']} d_dyn={a['d_dyn']} "
          f"enc_h={a['enc_hidden_ch']} dec_h={a['dec_hidden_ch']} use_ln={use_ln} "
          f"pool={pool_type} dec={decoder_type} lambda_mae={a.get('lambda_mae', 0)}\n")

    seed = int(a.get("seed", 42)); mv = int(a.get("max_videos", 0))
    if args.val_frac > 0:
        ds_tr = ClevrerChunkPairs(args.cache_dir, split="train", val_frac=args.val_frac,
                                  seed=seed, max_videos=mv)
        ds_va = ClevrerChunkPairs(args.cache_dir, split="val", val_frac=args.val_frac,
                                  seed=seed, max_videos=mv)
    else:
        ds_tr = ClevrerChunkPairs(args.cache_dir, seed=seed, max_videos=mv)
        ds_va = ClevrerChunkPairs(args.cache_dir, seed=seed, max_videos=0)  # superset = "unseen"

    idx_tr = one_idx_per_video(ds_tr, args.n_videos)
    idx_va = one_idx_per_video(ds_va, args.n_videos)

    # ---------------- A. activation health (hooks on every SiLU) -------------
    acts = defaultdict(list)

    def mk_hook(name):
        def hook(_m, _i, out):
            # channel axis = 1 for conv outputs
            o = out.detach()
            dims = tuple(d for d in range(o.dim()) if d != 1)
            acts[name].append((o.abs().mean(dim=dims).cpu(), o.std(dim=dims).cpu()))
        return hook

    handles = []
    for mod, prefix in ((enc, "enc"), (dec, "dec")):
        for n, m in mod.named_modules():
            if isinstance(m, torch.nn.SiLU):
                handles.append(m.register_forward_hook(mk_hook(f"{prefix}.{n}")))

    batch = torch.stack([ds_tr[i]["chunk_obs"] for i in idx_tr[:64]]).to(device)
    with torch.no_grad():
        o = enc(batch); _ = dec(o["z_static"], o["z_dyn"])
    for h in handles:
        h.remove()

    print("=== A. activation health (64 train chunks) ===")
    print(f"{'layer':>28} {'ch':>5} {'mean|act|':>10} {'inert<1e-3':>11} {'inert<1e-2':>11}")
    for name, rec in acts.items():
        m = torch.stack([r[0] for r in rec]).mean(0)
        s = torch.stack([r[1] for r in rec]).mean(0)
        print(f"{name:>28} {len(m):>5} {m.mean():>10.4f} "
              f"{(s < 1e-3).float().mean():>10.1%} {(s < 1e-2).float().mean():>10.1%}")

    # ---------------- B. latent geometry, train vs val -----------------------
    print("\n=== B. latent geometry ===")
    rows = {}
    for tag, ds, idxs in (("train", ds_tr, idx_tr), ("val", ds_va, idx_va)):
        zs, zd = collect_z(enc, ds, idxs, device)
        pr_s, eig_s = participation_ratio(zs)
        zd_flat = zd.reshape(-1, zd.shape[-1])
        pr_d, eig_d = participation_ratio(zd_flat)
        dt = np.linalg.norm(np.diff(zd, axis=1), axis=-1).mean()
        rows[tag] = (zs, zd)
        print(f"[{tag}] z_static: per-dim std mean={zs.std(0).mean():.4f} "
              f"min={zs.std(0).min():.4f} max={zs.std(0).max():.4f} | "
              f"PR={pr_s:.1f}/{zs.shape[1]} | pairwise-cos={pairwise_cos(zs):.3f} | "
              f"top5 eig share={eig_s[:5].sum() / max(eig_s.sum(), 1e-12):.2f}")
        print(f"[{tag}] z_dyn   : per-dim std mean={zd_flat.std(0).mean():.4f} | "
              f"PR={pr_d:.1f}/{zd.shape[-1]} | pairwise-cos={pairwise_cos(zd_flat):.3f} | "
              f"mean ||z_dyn[t+1]-z_dyn[t]||={dt:.3f} vs ||z_dyn||={np.linalg.norm(zd_flat, axis=1).mean():.3f}")
    # cross-set geometry: is val inside the train manifold?
    zs_tr, _ = rows["train"]; zs_va, _ = rows["val"]
    c_tr = zs_tr.mean(0)
    d_tr = np.linalg.norm(zs_tr - c_tr, axis=1).mean()
    d_va = np.linalg.norm(zs_va - c_tr, axis=1).mean()
    print(f"[cross] mean dist to TRAIN centroid: train={d_tr:.3f} val={d_va:.3f} "
          f"(val/train={d_va / max(d_tr, 1e-9):.2f}; <1 means val codes huddle at the centroid)")

    # ---------------- C. decoder per-dim sensitivity -------------------------
    print("\n=== C. decoder dim usage (output RMS change per +1sigma_train poke) ===")
    n_probe = min(32, len(idx_va))
    b = torch.stack([ds_va[i]["chunk_obs"] for i in idx_va[:n_probe]]).to(device)
    with torch.no_grad():
        o = enc(b); zs0, zd0 = o["z_static"], o["z_dyn"]
        base = dec(zs0, zd0)
        sig_s = torch.tensor(zs_tr.std(0), device=device, dtype=zs0.dtype)
        sens_s = []
        for i in range(zs0.shape[1]):
            z = zs0.clone(); z[:, i] += sig_s[i]
            sens_s.append((dec(z, zd0) - base).pow(2).mean().sqrt().item())
        zd_std = torch.tensor(rows["train"][1].reshape(-1, zd0.shape[-1]).std(0),
                              device=device, dtype=zd0.dtype)
        sens_d = []
        for i in range(zd0.shape[-1]):
            z = zd0.clone(); z[:, :, i] += zd_std[i]
            sens_d.append((dec(zs0, z) - base).pow(2).mean().sqrt().item())
    for nm, s in (("z_static", np.array(sens_s)), ("z_dyn", np.array(sens_d))):
        med = np.median(s)
        print(f"{nm}: median dRMS={med:.4f} | dims <10% of median: "
              f"{(s < 0.1 * med).sum()}/{len(s)} | dims <30%: {(s < 0.3 * med).sum()}/{len(s)} | "
              f"top3 dims {np.argsort(s)[-3:][::-1].tolist()} carry "
              f"{np.sort(s)[-3:].sum() / max(s.sum(), 1e-12):.1%} of total sensitivity")

    # ---------------- D. pooling bottleneck ----------------------------------
    print("\n=== D. encoder pooling bottleneck (dyn trunk, 64 train chunks) ===")
    with torch.no_grad():
        h = enc.trunk_dyn(batch)                      # (B, C, T, H, W)
        cell = h.permute(0, 2, 3, 4, 1).reshape(len(batch), -1, h.shape[1])  # (B, 576, C)
        pooled = cell.mean(1)                          # what the head sees (per chunk, t-avg ok)
        spread = (cell - pooled.unsqueeze(1)).norm(dim=-1).mean()
        sal = batch.var(dim=1).reshape(len(batch), -1)              # (B, 576) input saliency
        thr = sal.quantile(0.75, dim=1, keepdim=True)
        obj = sal >= thr
        f_obj = cell[obj.unsqueeze(-1).expand_as(cell)].view(-1, h.shape[1])
        f_bg = cell[(~obj).unsqueeze(-1).expand_as(cell)].view(-1, h.shape[1])
        print(f"pre-pool cell-feature spread (mean ||cell - pooled||): {spread:.3f} "
              f"vs pooled norm {pooled.norm(dim=-1).mean():.3f}")
        print(f"object-cell vs background-cell mean feature distance: "
              f"{(f_obj.mean(0) - f_bg.mean(0)).norm():.3f} "
              f"(0 => trunk does not separate objects from background)")
        # what fraction of cells dominate the mean? top-k cell contribution
        cn = cell.norm(dim=-1)
        topk = cn.topk(58, dim=1).values.sum(1) / cn.sum(1)
        print(f"top-10% cells' share of total cell-feature norm: {topk.mean():.1%} "
              f"(~10% => norms uniform; mean-pool dilutes objects 10:1)")

    print("\n[done]")


if __name__ == "__main__":
    main()
