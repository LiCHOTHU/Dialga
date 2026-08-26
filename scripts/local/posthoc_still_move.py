"""Re-score every sweep arm on the stationary-vs-moving question, identically.

The in-training probe is computed with whatever the dataset happened to expose at
the time, so arms trained before and after a metric fix are not comparable. This
reloads each arm's checkpoint and recomputes the split under ONE definition, which
also lets us ask the sharper question the in-training version cannot:

  presence   which attribute classes appear among stationary (resp. moving) objects.
             Weak, because a colour often appears in BOTH groups, so the two label
             sets are correlated and a code that knows "red is here somewhere" scores
             on both.
  exclusive  classes present in one group and ABSENT from the other. This is the
             clean test of the original intuition: if z_static is the non-moving
             scene, it should read exclusively-stationary attributes and NOT
             exclusively-moving ones, and z_dyn should do the reverse.

Speed is measured over the whole video (max per-chunk mean speed per object).
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
from src.model.camera_pose import synthetic_pan_sequence       # noqa: E402
from src.model.memory_encoder import MemoryEncoder             # noqa: E402
from scripts.local.train_memory import probe                   # noqa: E402


@torch.no_grad()
def features(ckpt_path: Path, loader, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    enc = MemoryEncoder(hidden_ch=a["enc_hidden_ch"], d_static=a["d_static"],
                        static_grid=a["static_grid"], d_dyn=a["d_dyn"],
                        dyn_grid=a["dyn_grid"], mem_update=a["mem_update"],
                        mem_collapse=a["mem_collapse"], d_pose=a["d_pose"],
                        zero_mean_dyn=a.get("zero_mean_dyn", False)).to(dev)
    enc.load_state_dict(ck["enc"])
    enc.eval()
    gen = torch.Generator(device=dev).manual_seed(1234)
    ZS, ZD, ATT, MASK, SPD = [], [], [], [], []
    for b in loader:
        seq = b["latents"].to(dev)
        pose = None
        if a["synth_pan"]:
            seq, pose = synthetic_pan_sequence(seq, generator=gen)
            if a["d_pose"] <= 0:
                pose = None
        grids, zdyn, _ = enc(seq, pose)
        ZS.append(grids[:, -1].flatten(1).cpu())
        ZD.append(zdyn.mean(dim=(1, 2)).cpu())
        ATT.append(b["attrs"]); MASK.append(b["slot_mask"]); SPD.append(b["speeds"])
    return (torch.cat(ZS).numpy(), torch.cat(ZD).numpy(), torch.cat(ATT).numpy(),
            torch.cat(MASK).numpy(), torch.cat(SPD).numpy(), a)


def score(ZS, ZD, ATT, MASK, SPD):
    sp = SPD[MASK]
    lo, hi = np.percentile(sp, 40), np.percentile(sp, 60)
    still, move = (SPD <= lo) & MASK, (SPD >= hi) & MASK

    def pres(sel):
        return ((ATT * sel[..., None]).max(axis=1) > 0.5).astype(np.float32), sel.sum(1)

    p_still, n_still = pres(still)
    p_move, n_move = pres(move)
    out = {}
    for tag, y, n in (("still", p_still, n_still), ("move", p_move, n_move)):
        ok = n > 0
        out[f"{tag}_zs"] = probe(ZS, y, ok)
        out[f"{tag}_zd"] = probe(ZD, y, ok)
    # exclusive: present in one group, absent from the other
    ex_still = ((p_still > 0.5) & (p_move < 0.5)).astype(np.float32)
    ex_move = ((p_move > 0.5) & (p_still < 0.5)).astype(np.float32)
    ok = (n_still > 0) & (n_move > 0)
    out["exstill_zs"] = probe(ZS, ex_still, ok)
    out["exstill_zd"] = probe(ZD, ex_still, ok)
    out["exmove_zs"] = probe(ZS, ex_move, ok)
    out["exmove_zd"] = probe(ZD, ex_move, ok)
    out["n"] = int(ok.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="outputs/mem_sweep")
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="outputs/logs/posthoc_still_move.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    va = ClevrerSequence(args.cache_dir, 4, 0, "val", preload=True)
    dl = DataLoader(va, batch_size=args.batch_size)
    print(f"[data] {len(va)} val videos")

    res = {}
    for d in sorted(Path(args.sweep_dir).iterdir()):
        ck = d / "ckpt.pt"
        if not ck.is_file():
            continue
        ZS, ZD, ATT, MASK, SPD, a = features(ck, dl, dev)
        res[d.name] = score(ZS, ZD, ATT, MASK, SPD)
        print(f"[done] {d.name}", flush=True)

    hdr = (f"{'arm':<20}{'still_zs':>9}{'still_zd':>9}{'move_zs':>9}{'move_zd':>9}"
           f"{'exSTILL_zs':>11}{'exSTILL_zd':>11}{'exMOVE_zs':>10}{'exMOVE_zd':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, v in res.items():
        print(f"{k:<20}{v['still_zs']:>9.3f}{v['still_zd']:>9.3f}"
              f"{v['move_zs']:>9.3f}{v['move_zd']:>9.3f}"
              f"{v['exstill_zs']:>11.3f}{v['exstill_zd']:>11.3f}"
              f"{v['exmove_zs']:>10.3f}{v['exmove_zd']:>10.3f}")
    print("\nThe intuition predicts exSTILL_zs > exSTILL_zd AND exMOVE_zd > exMOVE_zs.")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[saved] {args.out}")
    print("POSTHOC_DONE")


if __name__ == "__main__":
    main()
