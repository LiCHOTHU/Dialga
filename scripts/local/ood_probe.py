"""Table: robustness to an APPEARANCE SHORTCUT -- the capability test.

Why this experiment exists. Our ablations show each code is necessary and CKA shows
they overlap less than an entangled control, but both are statistics about the code.
The swap test, which is the field's usual capability claim, turned out to be passed
equally by a shared-trunk encoder that cannot factorize (sec:q2), so we have no
capability result. This is one, and it is not fakeable:

  Train a probe to predict MOTION while appearance is spuriously correlated with the
  label. At test time, break the correlation.

A code with appearance factored OUT is unaffected -- it never had access to the
shortcut. An entangled code, or one where identity leaked into the motion half, learned
the shortcut and collapses when it is removed. The gap between train-time and
shifted-test accuracy is therefore a direct read on whether appearance really is absent
from z_dyn, and no amount of decoder wiring can produce it artificially.

Implementation. On CLEVRER we have ground-truth colour and per-object motion. We form a
binary motion target (is the fastest object moving faster than the median clip?) and
construct a training set in which colour predicts that target with probability `rho`,
then a test set where the association is inverted. Reported: accuracy on an i.i.d. test
split, accuracy on the shifted split, and the drop.
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
from src.data.clevrer_sequence import ClevrerSequence               # noqa: E402
from scripts.local.eval_psnr import build                            # noqa: E402


@torch.no_grad()
def gather(ckpt, loader, dev):
    enc, _, _ = build(ckpt, dev)
    ZS, ZD, SPD, ATT, MASK = [], [], [], [], []
    for b in loader:
        g, z, _ = enc(b["latents"].to(dev))
        ZS.append(g[:, 0].flatten(1).cpu())
        ZD.append(z[:, 0].mean(1).cpu())
        SPD.append(b["speeds"]); ATT.append(b["attrs"]); MASK.append(b["slot_mask"])
    return (torch.cat(ZS).numpy(), torch.cat(ZD).numpy(), torch.cat(SPD).numpy(),
            torch.cat(ATT).numpy(), torch.cat(MASK).numpy())


def fit_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=800).fit(sc.transform(Xtr), ytr)
    return float((m.predict(sc.transform(Xte)) == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--rho", type=float, default=0.9,
                    help="P(shortcut colour present | motion label = 1) at TRAIN time")
    ap.add_argument("--max_videos", type=int, default=1500)
    ap.add_argument("--out", default="outputs/logs/ood_probe.json")
    args = ap.parse_args()
    dev = torch.device("cuda")

    ds = ClevrerSequence(args.cache_dir, 4, args.max_videos, "train", preload=True)
    dl = DataLoader(ds, batch_size=32)
    print(f"[data] {len(ds)} clips", flush=True)

    res = {}
    print(f"\n{'model':<24}{'feature':<10}{'i.i.d.':>9}{'shifted':>9}{'drop':>8}")
    print('-' * 62)
    for ck, lb in zip(args.ckpts, args.labels):
        ZS, ZD, SPD, ATT, MASK = gather(ck, dl, dev)
        # motion target: does this clip contain a fast object?
        # Motion target. A median split gives a label whose hardest examples sit right
        # at the boundary, and the probe then reads ~0.55 on every model -- too weak to
        # lose anything under shift. Use the extreme terciles and drop the ambiguous
        # middle, so the i.i.d. task is actually learnable and a collapse is visible.
        fast = np.where(MASK, SPD, 0).max(1)
        lo, hi = np.quantile(fast, [0.33, 0.67])
        clear = (fast <= lo) | (fast >= hi)
        y = (fast >= hi).astype(int)
        # shortcut variable: presence of colour 0 among real objects
        col0 = ((ATT[:, :, 0] * MASK) > 0.5).any(1).astype(int)
        rng = np.random.RandomState(0)
        # TRAIN pool: keep clips where colour agrees with the label w.p. rho
        agree = (col0 == y)
        keep = np.where(agree, rng.rand(len(y)) < args.rho, rng.rand(len(y)) < 1 - args.rho)
        keep &= clear                      # only unambiguous-motion clips
        idx = np.where(keep)[0]; rng.shuffle(idx)
        n = int(len(idx) * .7)
        tr, iid = idx[:n], idx[n:]
        # SHIFTED test: clips where colour DISagrees with the label
        shift = np.where((~agree) & clear)[0]
        if len(shift) < 50 or len(tr) < 50:
            print(f"{lb:<24} insufficient clips (train {len(tr)}, shift {len(shift)})")
            continue
        r = {}
        for nm, X in (("z_static", ZS), ("z_dyn", ZD)):
            a_iid = fit_eval(X[tr], y[tr], X[iid], y[iid])
            a_sh = fit_eval(X[tr], y[tr], X[shift], y[shift])
            r[nm] = {"iid": a_iid, "shifted": a_sh, "drop": a_iid - a_sh}
            print(f"{lb:<24}{nm:<10}{a_iid:>9.3f}{a_sh:>9.3f}{a_iid-a_sh:>+8.3f}", flush=True)
        res[lb] = r
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("\nA z_dyn with appearance factored OUT should show a small drop; a code that "
          "\nabsorbed appearance learns the colour shortcut and falls when it inverts.")
    print("OOD_PROBE_OK")


if __name__ == "__main__":
    main()
