"""SSv2 frozen-feature action-recognition probe (paper Table tab:ssv2).

Freezes a trained DIALGA encoder and fits a linear classifier on frozen chunk
features to predict the 174-way SSv2 action class. The controlled comparison is
WHICH SLOT carries the motion signal:

    zdyn     : mean-over-time of z_dyn  (the dynamics code -- expected best)
    zstatic  : z_static                 (appearance only -- expected weak)
    both     : concat(zstatic, zdyn)
    wanmean  : raw Wan latent mean-pool (appearance-only reference)
    random   : z_dyn from a RANDOM-INIT encoder (same arch, untrained)

Features come from the per-clip Wan-latent caches built by cache_wan_ssv2.py
(each blob carries label_id). Train the probe on the train cache, evaluate on a
disjoint val cache. One clip = 3 windows; we average the 3 window features into
one clip feature (test-time) and treat each window as a training example.

Usage:
    python scripts/probes/ssv2_action_probe.py \\
        --ckpt .../ssv2_train/v5_best.pt \\
        --train_cache .../ssv2_8000vid_W33 \\
        --val_cache   .../ssv2_val2000_W33
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.latent_encoder import LatentEncoder3D


def build_encoder(ckpt, device):
    a = ckpt.get("args", {})
    def g(k, d):
        return a[k] if k in a else d
    use_ln = "norm_static.weight" in ckpt["encoder"]
    d_pose = int(g("d_pose", 0)) if g("use_camera_pose", False) else 0
    enc = LatentEncoder3D(
        d_static=int(g("d_static", 96)), d_dyn=int(g("d_dyn", 256)),
        hidden_ch=int(g("enc_hidden_ch", 192)),
        shared_trunk=bool(g("shared_trunk", False)),
        pool_type=g("pool_type", "spatial"),
        n_queries=int(g("pool_queries", 8)), n_heads=int(g("pool_heads", 4)),
        static_grid=int(g("static_grid", 4)),
        chunk_size_lat=int(g("chunk_size_lat", 9)),
        static_agg=g("static_agg", "conv"),
        dyn_spatial=bool(g("dyn_spatial", True)), dyn_grid=int(g("dyn_grid", 8)),
        d_pose=d_pose,
    ).to(device)
    return enc, use_ln


@torch.no_grad()
def extract(cache_dir, enc, device, want_random_enc=None):
    """Return dict of per-window feature matrices + labels for one cache."""
    meta = json.loads((Path(cache_dir) / "metadata.json").read_text())
    feats = {k: [] for k in ["zstatic", "zdyn", "wanmean", "wanflat", "random"]}
    labels, vids = [], []
    for w in meta["windows"]:
        blob = torch.load(Path(cache_dir) / w["path"], map_location="cpu", weights_only=False)
        lat = blob["latent"].unsqueeze(0).to(device)             # (1,C,T,H,W)
        out = enc(lat)
        feats["zstatic"].append(out["z_static"][0].float().cpu().numpy())
        zd = out["z_dyn"][0]                                      # (T, D_d)
        feats["zdyn"].append(zd.float().mean(0).cpu().numpy())
        feats["wanmean"].append(blob["latent"].mean(dim=(1, 2, 3)).numpy())
        feats["wanflat"].append(blob["latent"].flatten().numpy())   # 27648-d = input ceiling
        if want_random_enc is not None:
            ro = want_random_enc(lat)
            feats["random"].append(ro["z_dyn"][0].float().mean(0).cpu().numpy())
        labels.append(int(blob.get("label_id", -1)))
        vids.append(int(blob["video_id"]))
    out = {k: np.stack(v) for k, v in feats.items() if v}
    out["both"] = np.concatenate([out["zstatic"], out["zdyn"]], axis=1)
    out["_labels"] = np.array(labels)
    out["_vids"] = np.array(vids)
    return out


def clip_pool(feat, vids):
    """Average window features within the same clip -> one row per clip."""
    uv = np.unique(vids)
    return (np.stack([feat[vids == v].mean(0) for v in uv]),
            np.array([vids == v for v in uv]))


def probe(train_X, train_y, val_X, val_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train_X)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(train_X), train_y)
    proba = clf.predict_proba(sc.transform(val_X))
    classes = clf.classes_
    top1 = (classes[proba.argmax(1)] == val_y).mean()
    top5idx = np.argsort(-proba, axis=1)[:, :5]
    top5 = np.mean([val_y[i] in classes[top5idx[i]] for i in range(len(val_y))])
    return top1 * 100, top5 * 100


def probe_mlp(train_X, train_y, val_X, val_y, device, hidden=1024, epochs=80, bs=256):
    """Nonlinear (1-hidden-layer MLP) probe. Tests whether info is present in the
    features but NON-LINEARLY encoded (a reconstruction-optimized latent packs
    semantics non-linearly, so a linear probe understates the accessible ceiling)."""
    import torch.nn as nn
    mu = train_X.mean(0, keepdims=True); sd = train_X.std(0, keepdims=True) + 1e-6
    classes = np.unique(train_y); cmap = {c: i for i, c in enumerate(classes)}
    Xtr = torch.tensor((train_X - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor([cmap[c] for c in train_y], dtype=torch.long)
    Xva = torch.tensor((val_X - mu) / sd, dtype=torch.float32).to(device)
    net = nn.Sequential(nn.Linear(train_X.shape[1], hidden), nn.GELU(),
                        nn.Dropout(0.5), nn.Linear(hidden, len(classes))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xtr[idx].to(device))
            loss = lossf(out, ytr[idx].to(device))
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(Xva).cpu().numpy()
    top1 = (classes[logits.argmax(1)] == val_y).mean()
    top5idx = np.argsort(-logits, axis=1)[:, :5]
    top5 = np.mean([val_y[i] in classes[top5idx[i]] for i in range(len(val_y))])
    return top1 * 100, top5 * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_cache", required=True)
    ap.add_argument("--val_cache", required=True)
    ap.add_argument("--mlp", action="store_true",
                    help="also run a nonlinear MLP probe (accessible-ceiling test)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc, use_ln = build_encoder(ckpt, device)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    rnd, _ = build_encoder(ckpt, device); rnd.eval()   # random-init, same arch
    print(f"[model] {args.ckpt}  use_ln={use_ln}")

    t0 = time.time()
    tr = extract(args.train_cache, enc, device, want_random_enc=rnd)
    va = extract(args.val_cache, enc, device, want_random_enc=rnd)
    print(f"[extract] train={len(tr['_labels'])} val={len(va['_labels'])} windows "
          f"in {time.time()-t0:.0f}s")

    # keep only labeled examples
    def clean(d):
        m = d["_labels"] >= 0
        return {k: (v[m] if k != "_vids" else v[m]) for k, v in d.items()}
    tr, va = clean(tr), clean(va)
    n_cls = len(np.unique(np.concatenate([tr["_labels"], va["_labels"]])))
    chance = 100.0 / n_cls
    print(f"[probe] {n_cls} classes present; chance top1={chance:.2f}%\n")

    rows = [("random", "z_dyn (rand-init)"), ("wanmean", "raw Wan mean"),
            ("wanflat", "raw Wan FULL (ceiling)"),
            ("zstatic", "z_static"), ("zdyn", "z_dyn"), ("both", "z_static+z_dyn")]
    hdr = f"{'feature':22s} {'dim':>5s} {'lin_top1':>9s} {'lin_top5':>9s}"
    if args.mlp:
        hdr += f" {'mlp_top1':>9s} {'mlp_top5':>9s}"
    print(hdr); print("-" * len(hdr))
    results = {}
    for key, name in rows:
        # val features pooled to one row per clip; train uses per-window rows
        val_feat, _ = clip_pool(va[key], va["_vids"])
        val_lab = np.array([va["_labels"][va["_vids"] == v][0]
                            for v in np.unique(va["_vids"])])
        t1, t5 = probe(tr[key], tr["_labels"], val_feat, val_lab)
        line = f"{name:22s} {tr[key].shape[1]:5d} {t1:8.2f}% {t5:8.2f}%"
        rec = {"dim": tr[key].shape[1], "lin_top1": round(t1, 2), "lin_top5": round(t5, 2)}
        if args.mlp:
            m1, m5 = probe_mlp(tr[key], tr["_labels"], val_feat, val_lab, device)
            line += f" {m1:8.2f}% {m5:8.2f}%"
            rec["mlp_top1"] = round(m1, 2); rec["mlp_top5"] = round(m5, 2)
        results[key] = rec
        print(line, flush=True)
    print(f"\n[chance] top1={chance:.2f}%  top5={5*chance:.2f}%")
    print("[json] " + json.dumps(results))
    print("SSV2_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
