"""Is a computed 'static scene' mosaic a good TARGET to guide z_static toward?

Zero training. Builds the explicit memory the way an explicit method would -- pool all
K*T latent frames of a video, robustly -- and asks two questions before any run is
spent on it:

  CEILING   how much of each chunk does that one video-level image explain? This is
            the cap that supervising z_static toward it would impose. If the mosaic
            explains little, the target is a bad teacher and the aux loss would drag
            z_static down rather than up.
  SELECTIVITY does a MEDIAN mosaic actually reject movers where a MEAN mosaic keeps
            them? That is the whole claim of the robust accumulation, and it decides
            whether the target encodes "the non-moving scene" or just "the average
            frame".

Compares the video-level target against the per-chunk one, since the gap between them
is exactly what a video-level z_static gives up.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def rel(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--max_videos", type=int, default=200)
    ap.add_argument("--out", default="outputs/logs/diag_static_target.json")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    meta = json.loads((cache / "metadata.json").read_text())
    rows = meta["windows"] if isinstance(meta, dict) else meta
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[int(r["video_id"])].append((int(r["start_frame"]), r["path"]))
    vids = [v for v, w in by_vid.items() if len(w) >= 4][: args.max_videos]

    res = defaultdict(list)
    for v in vids:
        xs = [torch.load(cache / p, map_location="cpu", weights_only=False)["latent"].float()
              for _, p in sorted(by_vid[v])]
        X = torch.cat(xs, dim=1)                      # (C, K*T, H, W) whole video
        vid_mean = X.mean(1)
        vid_med = X.median(1).values

        for x in xs:                                   # (C,T,H,W) one chunk
            ch_mean = x.mean(1)
            res["chunk_mean_explains_chunk"].append(rel(ch_mean.unsqueeze(1), x))
            res["video_mean_explains_chunk"].append(rel(vid_mean.unsqueeze(1), x))
            res["video_med_explains_chunk"].append(rel(vid_med.unsqueeze(1), x))

        # selectivity: where does the video mosaic disagree with the per-chunk one,
        # and is that where motion is?
        var = X.var(1).mean(0)                         # (H,W) motion energy per cell
        thr = var.flatten().quantile(0.75)
        hi, lo = var >= thr, var < thr
        d = (vid_mean - vid_med).pow(2).mean(0)
        res["med_vs_mean_high_motion"].append(float(d[hi].mean()))
        res["med_vs_mean_low_motion"].append(float(d[lo].mean()))

    out = {k: float(sum(v) / len(v)) for k, v in res.items()}
    out["selectivity_ratio"] = (out["med_vs_mean_high_motion"]
                                / max(1e-12, out["med_vs_mean_low_motion"]))
    out["n_videos"] = len(vids)

    print(f"[data] {len(vids)} videos, {len(res['chunk_mean_explains_chunk'])} chunks\n")
    print("=== CEILING: fraction of a chunk NOT explained by a static image (lower=better)")
    print(f"  per-CHUNK  mean  : {out['chunk_mean_explains_chunk']:.4f}   "
          f"(what today's per-chunk z_static can aim at)")
    print(f"  per-VIDEO  mean  : {out['video_mean_explains_chunk']:.4f}")
    print(f"  per-VIDEO  median: {out['video_med_explains_chunk']:.4f}   "
          f"(the proposed teacher)")
    gap = out["video_med_explains_chunk"] - out["chunk_mean_explains_chunk"]
    print(f"  -> a video-level target gives up {gap:+.4f} vs a per-chunk one")
    print("\n=== SELECTIVITY: median-vs-mean disagreement, by cell type")
    print(f"  high-motion cells: {out['med_vs_mean_high_motion']:.5f}")
    print(f"  low-motion  cells: {out['med_vs_mean_low_motion']:.5f}")
    print(f"  ratio (want >>1) : {out['selectivity_ratio']:.2f}x")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}\nTARGET_DIAG_DONE")


if __name__ == "__main__":
    main()
