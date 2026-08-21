"""Leg-3 (predictable): is each frozen representation forward-predictable by a
learned dynamics model, beyond the trivial copy-last baseline?  Scale-free, fair.

Each CLEVRER video gives 3 time-ordered chunks (start_frame 0,33,66). For every
representation we extract per-chunk features [c0,c1,c2], then train ONE small MLP
forward model f: c_t -> c_{t+1} on train videos and evaluate on held-out videos.

Metric (scale-free, comparable across representations of any dim/scale):
  copy_mse  = mean || c_{t+1} - c_t ||^2   (predict "no change")
  pred_mse  = mean || c_{t+1} - f(c_t) ||^2
  skill     = 1 - pred_mse / copy_mse   (>0 means the learned model beats copy-last)
A "predictable" representation has skill clearly > 0. Our dynamics code z_dyn should;
appearance-only encoders (DINOv2) have little chunk-to-chunk motion to predict.

Usage:
  python scripts/probes/clevrer_rollout_baselines.py --ckpt .../ckpt.pt \
     --cache_dir .../wan_10000vid_W33 --video_root .../CLEVRER/train_video \
     --methods ours videomae videoflextok dinov2 --max_videos 2000 --out .../roll.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_ssv2 import read_clip
from scripts.probes.clevrer_baselines_probe import mp4_path, build_extractor
from scripts.probes.clevrer_decode_baselines import build_our_encoder
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate


@torch.no_grad()
def our_dyn(enc, x):
    return enc(x.unsqueeze(0))["z_dyn"].flatten(1)[0].cpu()   # z_dyn only (the claim)


def per_video_chunks(method, split, a, args, enc, device):
    """-> dict video_id -> tensor (3, D) ordered by start_frame (videos with 3 chunks).

    The pair sampler only ever exposes chunks at start_frame {0,33} as `obs`, so we
    read all 3 chunks straight from the cache metadata; the dataset is used only to
    reproduce the exact train/val video-id split.
    """
    ds = ClevrerChunkPairs(a["cache_dir"], split=split,
                           val_frac=float(a.get("val_frac", 0.1)),
                           seed=int(a.get("seed", 42)), max_videos=args.max_videos)
    split_vids = {int(ds.windows[p[0]]["video_id"]) for p in ds.pairs}
    by_vid: dict[int, dict[int, str]] = {}
    for w in ds.windows:
        by_vid.setdefault(int(w["video_id"]), {})[int(w["start_frame"])] = w["path"]
    cache_dir = Path(a["cache_dir"])

    ext = None if method == "ours" else build_extractor(method, device)
    out = {}
    for vid in sorted(split_vids):
        sfs = sorted(by_vid.get(vid, {}))[:3]
        if len(sfs) < 3:
            continue
        feats = []
        clip = None
        for sf in sfs:
            if method == "ours":
                blob = torch.load(cache_dir / by_vid[vid][sf],
                                  map_location="cpu", weights_only=False)
                feats.append(our_dyn(enc, blob["latent"].to(device)))
            else:
                if clip is None:
                    clip = read_clip(str(mp4_path(args.video_root, vid)))
                if clip is None:
                    break
                feats.append(torch.from_numpy(
                    ext.feat(clip, [sf], args.window_frames)).float())
        if len(feats) == 3:
            out[vid] = torch.stack(feats)                     # (3, D)
    if ext is not None:
        del ext; torch.cuda.empty_cache()
    return out


def make_pairs(vid2chunks):
    src, tgt = [], []
    for c in vid2chunks.values():                              # c: (3,D)
        src.append(c[0]); tgt.append(c[1])
        src.append(c[1]); tgt.append(c[2])
    return torch.stack(src), torch.stack(tgt)


def eval_method(method, a, args, enc, device):
    tr = per_video_chunks(method, "train", a, args, enc, device)
    va = per_video_chunks(method, "val", a, args, enc, device)
    Xtr, Ytr = make_pairs(tr)
    Xva, Yva = make_pairs(va)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xn, Yn = (Xtr - mu) / sd, (Ytr - mu) / sd
    Xvn, Yvn = (Xva - mu) / sd, (Yva - mu) / sd
    D = Xn.shape[1]
    fwd = nn.Sequential(nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, 1024),
                        nn.GELU(), nn.Linear(1024, D)).to(device)
    opt = torch.optim.AdamW(fwd.parameters(), lr=3e-4, weight_decay=1e-4)
    Xd, Yd = Xn.to(device), Yn.to(device)
    n, bs = Xn.shape[0], 128
    for ep in range(args.epochs):
        fwd.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(fwd(Xd[idx]), Yd[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    fwd.eval()
    with torch.no_grad():
        pred = fwd(Xvn.to(device)).cpu()
    pred_mse = F.mse_loss(pred, Yvn).item()
    copy_mse = F.mse_loss(Xvn, Yvn).item()                     # predict "no change"
    return {"dim": int(D), "n_val_pairs": int(Xva.shape[0]),
            "pred_mse": round(pred_mse, 5), "copy_mse": round(copy_mse, 5),
            "skill": round(1.0 - pred_mse / max(copy_mse, 1e-9), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["ours", "videomae", "videoflextok", "dinov2"])
    ap.add_argument("--max_videos", type=int, default=2000)
    ap.add_argument("--window_frames", type=int, default=33)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
    enc = build_our_encoder(a, ck["encoder"], device)

    results = {}
    for m in args.methods:
        t0 = time.time()
        results[m] = eval_method(m, a, args, enc, device)
        r = results[m]
        print(f"[{m}] dim={r['dim']} pred_mse={r['pred_mse']} copy_mse={r['copy_mse']} "
              f"skill={r['skill']} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\nROLLOUT_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
