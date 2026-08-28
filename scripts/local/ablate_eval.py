"""Table: how much each code actually matters to the decoder, on ANY dataset.

The number we report as the headline of Q1/Q4 is a DELETION COST: zero one code, and
measure how far reconstruction degrades. It is the only statistic that distinguishes a
static code the decoder depends on from one it ignores, and it is what separates the
base+delta wiring from the free decoder (11% -> 484% on CLEVRER).

This exists as a standalone script because the equivalent lived only inside the
trainer's eval loop, and `swap_eval.py` -- the other route to it -- computes an identity
metric from CLEVRER attributes and so cannot run on SSv2 or LIBERO at all. Nothing here
touches labels, so it runs on any cache.

Reported per model:
  full        reconstruction MSE with both codes
  no_static   z_static zeroed          -> cost as % increase over `full`
  no_dyn      z_dyn zeroed             -> cost as % increase over `full`
  swap        z_static taken from a DIFFERENT video (batch roll)

A working split needs BOTH deletion costs to be large. One large and one small means the
big code carries the video and the other is decorative.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local.eval_psnr import build                          # noqa: E402


def loader(cache_dir, n, dataset=None):
    nm = Path(cache_dir).name.lower()
    ds = dataset or ("ssv2" if ("ssv2" in nm or "libero" in nm) else "clevrer")
    if ds == "ssv2":
        from src.data.ssv2_sequence import SSv2Sequence
        return SSv2Sequence(cache_dir, 4, n, "val", preload=True)
    from src.data.clevrer_sequence import ClevrerSequence
    return ClevrerSequence(cache_dir, 4, n, "val", preload=True)


@torch.no_grad()
def run(ckpt, dl, dev):
    enc, dec, _ = build(ckpt, dev)
    acc = {}
    for b in dl:
        seq = b["latents"].to(dev)
        g, zd, _ = enc(seq)
        k = 0
        tgt = seq[:, k]
        acc.setdefault("full", []).append(float(F.mse_loss(dec(g[:, k], zd[:, k]), tgt)))
        acc.setdefault("no_static", []).append(
            float(F.mse_loss(dec(torch.zeros_like(g[:, k]), zd[:, k]), tgt)))
        acc.setdefault("no_dyn", []).append(
            float(F.mse_loss(dec(g[:, k], torch.zeros_like(zd[:, k])), tgt)))
        acc.setdefault("swap", []).append(
            float(F.mse_loss(dec(g[:, k].roll(1, 0), zd[:, k]), tgt)))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    f = m["full"]
    m["zs_cost_pct"] = 100.0 * (m["no_static"] - f) / f
    m["zd_cost_pct"] = 100.0 * (m["no_dyn"] - f) / f
    m["swap_cost_pct"] = 100.0 * (m["swap"] - f) / f
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--cache_dir", default="outputs/cache/ssv2_W17_full")
    ap.add_argument("--dataset", choices=["clevrer", "ssv2"], default=None)
    ap.add_argument("--n_videos", type=int, default=400)
    ap.add_argument("--out", default="outputs/logs/ablate.json")
    args = ap.parse_args()
    dev = torch.device("cuda")

    va = loader(args.cache_dir, args.n_videos, args.dataset)
    dl = DataLoader(va, batch_size=32)
    print(f"[data] {len(va)} val videos\n")
    print(f"{'model':<14}{'full MSE':>10}{'del z_static':>14}{'del z_dyn':>12}{'swap':>10}")
    print('-' * 60)
    res = {}
    for ck, lb in zip(args.ckpts, args.labels):
        r = run(ck, dl, dev); res[lb] = r
        print(f"{lb:<14}{r['full']:>10.5f}{r['zs_cost_pct']:>13.0f}%"
              f"{r['zd_cost_pct']:>11.0f}%{r['swap_cost_pct']:>9.0f}%", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\nBoth deletion costs must be large. One large and one small means the big "
          "\ncode carries the video and the other is decorative.")
    print("ABLATE_OK")


if __name__ == "__main__":
    main()
