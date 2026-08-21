"""Formal disentanglement metrics on CLEVRER (paper Table tab:q2_dci).

DCI (Disentanglement / Completeness / Informativeness), MIG (Mutual Information
Gap), and SAP, computed for a trained DIALGA checkpoint's frozen codes against
CLEVRER ground-truth attributes.

IMPORTANT methodological note. Textbook DCI/MIG assume ONE factor value per
sample (dSprites, Shapes3D). CLEVRER scenes are MULTI-OBJECT, so there is no
single (colour, material, shape) per clip. We therefore use the
*attribute-presence* adaptation: each of the 13 attribute values
(8 colours + 2 materials + 3 shapes) is a BINARY factor "is an object with this
value present in the scene". Metrics are computed over these 13 binary factors.
This measures whether the latent dimensions specialise to individual attributes;
it is an adaptation, not the single-object benchmark, and is reported as such.

Codes compared: z_static (96), z_dyn (mean-over-time, 256), and [z_static+z_dyn].
A disentangled identity code should score high D/C/MIG on colour/material/shape
via z_static and near-zero via z_dyn (the cross-probe asymmetry, quantified).

Usage:
  python scripts/probes/factorization_dci.py --ckpt .../v59_clevrer/v5_best.pt \\
      --cache_dir .../wan_10000vid_W33 --max_videos 1500 --out .../q2_dci.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.ssv2_action_probe import build_encoder

# CLEVRER attrs layout: 13 = [8 colour | 2 material | 3 shape]
COL, MAT, SHP = slice(0, 8), slice(8, 10), slice(10, 13)


@torch.no_grad()
def extract(cache_dir, enc, device, max_videos):
    meta = json.loads((Path(cache_dir) / "metadata.json").read_text())
    wins = meta["windows"]
    if max_videos:
        wins = wins[:max_videos]
    zs, zd, presence = [], [], []
    for w in wins:
        b = torch.load(Path(cache_dir) / w["path"], map_location="cpu", weights_only=False)
        lat = b["latent"].unsqueeze(0).to(device)
        o = enc(lat)
        zs.append(o["z_static"][0].float().cpu().numpy())
        zd.append(o["z_dyn"][0].float().mean(0).cpu().numpy())
        attrs = b["attrs"].float()                       # (K,13)
        mask = b["slot_mask"].bool()                     # (K,)
        a = attrs[mask]                                  # (n_obj,13)
        pres = (a > 0.5).any(0).numpy().astype(np.float32)  # (13,) present?
        presence.append(pres)
    return np.stack(zs), np.stack(zd), np.stack(presence)


def mig(codes, factors, n_bins=20):
    """Mean Mutual Information Gap over binary factors."""
    from sklearn.feature_selection import mutual_info_classif
    # discretise codes for MI estimate stability handled by mutual_info_classif (kNN)
    migs = []
    for j in range(factors.shape[1]):
        y = factors[:, j].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        mi = mutual_info_classif(codes, y, discrete_features=False, random_state=0)
        mi_sorted = np.sort(mi)[::-1]
        Hy = -np.mean([p * np.log(p + 1e-12) for p in
                       [y.mean(), 1 - y.mean()]])
        if Hy < 1e-6:
            continue
        migs.append((mi_sorted[0] - mi_sorted[1]) / Hy)
    return float(np.mean(migs)) if migs else 0.0


def dci(codes, factors):
    """DCI Disentanglement + Completeness + Informativeness via RF importances."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    F = factors.shape[1]
    D = codes.shape[1]
    R = np.zeros((F, D))
    infos = []
    for j in range(F):
        y = factors[:, j].astype(int)
        if len(np.unique(y)) < 2:
            continue
        Xtr, Xte, ytr, yte = train_test_split(codes, y, test_size=0.3, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        rf.fit(Xtr, ytr)
        R[j] = rf.feature_importances_
        infos.append(rf.score(Xte, yte))
    # normalise importance matrix
    Rf = R / (R.sum(1, keepdims=True) + 1e-12)      # per-factor over codes -> completeness
    Rc = R / (R.sum(0, keepdims=True) + 1e-12)      # per-code over factors -> disentanglement

    def ent(p, base):
        p = p[p > 0]
        return -np.sum(p * (np.log(p) / np.log(base))) if len(p) else 0.0

    # Disentanglement: 1 - entropy of each code's importance over factors, weighted by code use
    d_codes = np.array([1 - ent(Rc[:, k], F) for k in range(D)])
    rho = R.sum(0) / (R.sum() + 1e-12)
    disent = float(np.sum(d_codes * rho))
    # Completeness: 1 - entropy of each factor's importance over codes
    compl = float(np.mean([1 - ent(Rf[j], D) for j in range(F)]))
    info = float(np.mean(infos)) if infos else 0.0
    return disent, compl, info


def evaluate(name, codes, factors):
    d, c, i = dci(codes, factors)
    m = mig(codes, factors)
    print(f"{name:16s} dim={codes.shape[1]:4d}  DCI-D={d:.3f}  DCI-C={c:.3f}  "
          f"Info={i:.3f}  MIG={m:.3f}")
    return {"dim": codes.shape[1], "DCI_D": round(d, 3), "DCI_C": round(c, 3),
            "Info": round(i, 3), "MIG": round(m, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc, _ = build_encoder(ckpt, device)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    print(f"[model] {args.ckpt}")

    zs, zd, pres = extract(args.cache_dir, enc, device, args.max_videos)
    print(f"[data] {len(zs)} scenes; factor-presence rate "
          f"col={pres[:, COL].mean():.2f} mat={pres[:, MAT].mean():.2f} shp={pres[:, SHP].mean():.2f}")

    res = {"ckpt": args.ckpt, "n": len(zs),
           "note": "attribute-presence adaptation (multi-object CLEVRER)"}
    res["z_static"] = evaluate("z_static", zs, pres)
    res["z_dyn"] = evaluate("z_dyn", zd, pres)
    res["z_static+z_dyn"] = evaluate("z_static+z_dyn", np.concatenate([zs, zd], 1), pres)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
