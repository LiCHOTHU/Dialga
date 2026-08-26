"""Diagnostic: is there a stable, video-length "static part" in the Wan latent?

Zero training. Answers three questions the memory design depends on:

  Q1 ENERGY   -- how much of a chunk's latent is temporally constant vs changing?
                 (upper bound on what a static/dynamic split can buy)
  Q2 DRIFT    -- how well does chunk k's static estimate match chunk 0's, for the
                 SAME video? This is the "z_static is not video-length consistent"
                 problem stated numerically. Compared across three collapses:
                 mean / median / trimmed-mean.
  Q3 MOVERS   -- does a temporal MEDIAN reject moving content that a MEAN smears in?
                 Measured where it matters: the high-temporal-variance cells.

Reads the Wan latent cache directly (no model, no checkpoint).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def collapse(x: torch.Tensor, how: str) -> torch.Tensor:
    """x : (C, T, H, W) -> (C, H, W) temporal collapse."""
    if how == "mean":
        return x.mean(dim=1)
    if how == "median":
        return x.median(dim=1).values
    if how == "trimmed":  # drop min+max per cell, mean the rest
        s, _ = x.sort(dim=1)
        return s[:, 1:-1].mean(dim=1)
    raise ValueError(how)


def rel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scale-aware distance: ||a-b||^2 / ||b||^2."""
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    files = sorted((cache / "latents").glob("*.pt"))
    if not files:
        raise SystemExit(f"no latents under {cache}")

    # group windows by video, ordered by start_frame
    by_video: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        by_video[int(blob["video_id"])].append((int(blob["start_frame"]), f))
    vids = [v for v, ws in by_video.items() if len(ws) >= 2][: args.max_videos]
    print(f"[data] {len(files)} windows, {len(by_video)} videos, "
          f"{len(vids)} usable (>=2 chunks)")

    HOWS = ["mean", "median", "trimmed"]
    energy_static, energy_dyn = [], []
    drift = {h: defaultdict(list) for h in HOWS}       # how -> lag -> [rel_mse]
    mover_gap, mover_gap_still = [], []

    for vid in vids:
        ws = sorted(by_video[vid])
        xs = []
        for _, f in ws:
            blob = torch.load(f, map_location="cpu", weights_only=False)
            xs.append(blob["latent"].float())          # (C,T,H,W)

        for x in xs:
            mu = x.mean(dim=1, keepdim=True)
            energy_static.append(float((mu ** 2).sum()))
            energy_dyn.append(float(((x - mu) ** 2).sum()))

        # Q2: drift of the static estimate across chunks of the SAME video
        for how in HOWS:
            base = collapse(xs[0], how)
            for lag in range(1, len(xs)):
                drift[how][lag].append(rel_mse(collapse(xs[lag], how), base))

        # Q3: mean-vs-median disagreement, split by per-cell temporal variance.
        # High-variance cells are where things move; if the median is doing its job
        # it should disagree with the mean THERE and agree elsewhere.
        for x in xs:
            var = x.var(dim=1).mean(dim=0)             # (H,W) motion energy per cell
            thr = var.flatten().quantile(0.75)
            hi, lo = var >= thr, var < thr
            d = (collapse(x, "mean") - collapse(x, "median")).pow(2).mean(dim=0)
            if hi.any():
                mover_gap.append(float(d[hi].mean()))
            if lo.any():
                mover_gap_still.append(float(d[lo].mean()))

    def avg(v):
        return float(sum(v) / max(1, len(v)))

    es, ed = avg(energy_static), avg(energy_dyn)
    res = {
        "n_videos": len(vids),
        "n_windows": len(files),
        "Q1_energy": {
            "static_frac": es / (es + ed),
            "dynamic_frac": ed / (es + ed),
        },
        "Q2_drift_rel_mse": {
            h: {f"lag{k}": avg(v) for k, v in sorted(drift[h].items())} for h in HOWS
        },
        "Q3_mean_vs_median": {
            "high_motion_cells": avg(mover_gap),
            "low_motion_cells": avg(mover_gap_still),
            "ratio": avg(mover_gap) / max(1e-12, avg(mover_gap_still)),
        },
    }
    print("\n=== Q1 energy split (per chunk) ===")
    print(f"  temporally CONSTANT part : {res['Q1_energy']['static_frac']*100:5.2f}%")
    print(f"  temporally CHANGING part : {res['Q1_energy']['dynamic_frac']*100:5.2f}%")
    print("\n=== Q2 static-estimate drift across chunks (rel-MSE vs chunk 0, lower=stabler) ===")
    lags = sorted(drift["mean"].keys())
    print("  collapse  " + "  ".join(f"lag{k}" for k in lags))
    for h in HOWS:
        print(f"  {h:9s} " + "  ".join(f"{avg(drift[h][k]):.4f}" for k in lags))
    print("\n=== Q3 mean-vs-median disagreement by cell type ===")
    print(f"  high-motion cells : {res['Q3_mean_vs_median']['high_motion_cells']:.5f}")
    print(f"  low-motion  cells : {res['Q3_mean_vs_median']['low_motion_cells']:.5f}")
    print(f"  ratio (want >1)   : {res['Q3_mean_vs_median']['ratio']:.2f}x")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\n[saved] {args.out}")
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
