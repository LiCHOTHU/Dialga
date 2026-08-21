"""External video-encoder baselines on the CLEVRER semantic (presence-mAP) probe.

The paper's Leg-1 claim is "compressive AND semantic": at a tiny 96-float code our
z_static reads object attributes better than the alternatives a reviewer will name.
`baseline_probe_table.py` already covers the free baselines (wan-mean, wan-PCA,
wan-flat, random-init, DINOv2 patch-mean). This script adds the *strong pretrained
video encoders* the reviewer will also name -- VideoMAE and VideoFlexTok -- on the
IDENTICAL val split and IDENTICAL presence-mAP protocol, at full dim AND PCA-reduced
to our width (96), so the comparison is apples-to-apples and dim-matched.

Features are frozen (no fine-tuning). Each CLEVRER chunk = 33 RGB frames starting at
start_frame in {0,33,66}; we read those frames from the raw mp4 and hand them to the
same extractors used for UCF101/SSv2. video_id -> mp4 path uses CLEVRER's standard
1000-video subfolder layout.

Usage:
  python scripts/probes/clevrer_baselines_probe.py \
     --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \
     --video_root /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
     --models videomae videoflextok --max_videos 1500 \
     --out .../clevrer_baselines.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_ssv2 import read_clip
from scripts.probes.baseline_probe_table import (
    presence_labels, eval_scores, run_probe, GROUPS)
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from torch.utils.data import DataLoader


def mp4_path(video_root, vid):
    lo = (vid // 1000) * 1000
    sub = f"video_{lo:05d}-{lo+1000:05d}"
    return Path(video_root) / sub / f"video_{vid:05d}.mp4"


def build_extractor(name, device):
    from scripts.probes.ucf101_baselines_probe import (
        VideoMAEExtractor, VideoFlexTokExtractor)
    if name == "videomae":
        return VideoMAEExtractor(device)
    if name == "videoflextok":
        return VideoFlexTokExtractor(device)
    if name == "dinov2":
        from scripts.probes.ssv2_dinov2_probe import DINOv2Extractor
        return DINOv2Extractor(device)
    raise ValueError(name)


def collect_labels_and_windows(cache_dir, split, val_frac, seed, max_videos):
    """Walk the dataset once -> dedup (video_id,start_frame) with presence labels."""
    ds = ClevrerChunkPairs(cache_dir, split=split, val_frac=val_frac,
                           seed=seed, max_videos=max_videos)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=chunk_collate)
    seen, wins, labs = set(), [], []
    for b in dl:
        Y = presence_labels(b["attrs"], b["slot_mask"])          # (B, N_ALL)
        for j in range(len(b["video_id"])):
            key = (int(b["video_id"][j]), int(b["start_frame"][j]))
            if key in seen:
                continue
            seen.add(key)
            wins.append(key)
            labs.append(Y[j])
    return wins, torch.stack(labs)


@torch.no_grad()
def extract(ext, wins, video_root, W, tag):
    """One feature vector per (video_id,start_frame) window. Caches clip per video."""
    feats, ok = [], np.ones(len(wins), dtype=bool)
    clip_cache_id, clip = -1, None
    t0 = time.time()
    for i, (vid, sf) in enumerate(wins):
        if vid != clip_cache_id:
            clip = read_clip(str(mp4_path(video_root, vid)))
            clip_cache_id = vid
        if clip is None:
            feats.append(np.zeros(ext.dim, np.float32)); ok[i] = False; continue
        feats.append(ext.feat(clip, [sf], W))
        if (i + 1) % 300 == 0:
            print(f"  [{tag}] {i+1}/{len(wins)} {time.time()-t0:.0f}s", flush=True)
    return np.stack(feats), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--models", nargs="+", default=["videomae", "videoflextok"])
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_videos", type=int, default=1500)
    ap.add_argument("--window_frames", type=int, default=33)
    ap.add_argument("--pca_dims", type=int, nargs="*", default=[96])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tr_w, Ytr = collect_labels_and_windows(args.cache_dir, "train",
                                           args.val_frac, args.seed, args.max_videos)
    va_w, Yva = collect_labels_and_windows(args.cache_dir, "val",
                                           args.val_frac, args.seed, args.max_videos)
    print(f"[data] train windows={len(tr_w)} val windows={len(va_w)}", flush=True)
    W = args.window_frames

    prior = eval_scores(Ytr.mean(0, keepdim=True).expand_as(Yva), Yva)
    pm = float(np.nanmean([prior[g] for g, _, _ in GROUPS]))
    print(f"[prior] mean mAP={pm:.3f}", flush=True)

    from sklearn.decomposition import PCA
    results = {"prior": prior, "models": {}}
    for name in args.models:
        ext = build_extractor(name, device)
        print(f"[model] {name} dim={ext.dim}", flush=True)
        Xtr, otr = extract(ext, tr_w, args.video_root, W, f"{name}-tr")
        Xva, ova = extract(ext, va_w, args.video_root, W, f"{name}-va")
        Xtr, Ytr_m = Xtr[otr], Ytr[torch.from_numpy(otr)]
        Xva, Yva_m = Xva[ova], Yva[torch.from_numpy(ova)]
        del ext; torch.cuda.empty_cache()

        rows = {}
        for d in [Xtr.shape[1]] + [k for k in args.pca_dims if k < Xtr.shape[1]]:
            if d == Xtr.shape[1]:
                Zt, Zv = Xtr, Xva
            else:
                pca = PCA(n_components=d, random_state=0).fit(Xtr)
                Zt, Zv = pca.transform(Xtr), pca.transform(Xva)
            sc, m, pe = run_probe(torch.from_numpy(Zt).float(), Ytr_m,
                                  torch.from_numpy(Zv).float(), Yva_m,
                                  device, args.epochs, args.lr, args.weight_decay)
            rows[str(d)] = {"dim": d, "mAP": m, "mean": sc, "probe_ep": pe}
            print(f"  {name} @dim {d:<5} color={m['color']:.3f} "
                  f"material={m['material']:.3f} shape={m['shape']:.3f} "
                  f"mean={sc:.3f}", flush=True)
        results["models"][name] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\nCLEVRER_BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
