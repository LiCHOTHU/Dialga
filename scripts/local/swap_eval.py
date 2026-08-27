"""Swap test: the factorization evidence that is a CAPABILITY, not a statistic.

Everything measured so far is either a correlation (leakage mAP -- where a shared-trunk
entangled baseline came within 0.04) or an ablation (zs_cost -- strong, but true by
construction once base+delta wires the split in). Neither shows something an entangled
model cannot do.

The swap does. Take two clips, cross their codes, decode the hybrid, and ask what the
hybrid CONTAINS:

    decode(zs_A, zd_B)  ->  should carry A's identity and B's motion

An entangled code cannot produce a coherent hybrid: its "static" half carries motion
and its "dynamic" half carries identity, so crossing them yields a blend of both or
garbage. Reading the right factor out of each half is only possible if the split is
real. This is the field-standard protocol (DiViD's swap accuracy, and the S3VAE /
C-DSVAE line before it), so the numbers are comparable outside this repo.

Both factors are measurable on CLEVRER: attributes for identity, positions for motion.
Probes are fit on NORMAL reconstructions and applied to hybrids, so a probe cannot
learn to exploit whatever artefacts swapping introduces.

Reported per model:
  id_from_swap    identity of the hybrid matches A (the z_static donor).  higher better
  id_leak_B       identity of the hybrid matches B (the z_dyn donor).     lower  better
  motion_from_B   motion of the hybrid matches B.                          higher better
  swap_margin     id_from_swap - id_leak_B: does the hybrid take identity
                  from the code that is supposed to carry it?
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
from src.data.clevrer_sequence import ClevrerSequence              # noqa: E402
from src.model.base_delta_decoder import BaseDeltaDecoder           # noqa: E402
from src.model.latent_decoder import SpatialGridDecoder             # noqa: E402
from src.model.memory_encoder import MemoryEncoder                  # noqa: E402


def build(ck_path, dev):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    enc = MemoryEncoder(hidden_ch=a["enc_hidden_ch"], d_static=a["d_static"],
                        static_grid=a["static_grid"], d_dyn=a["d_dyn"],
                        dyn_grid=a["dyn_grid"], mem_update=a["mem_update"],
                        mem_collapse=a["mem_collapse"], d_pose=a["d_pose"],
                        chunk_size_lat=a.get("chunk_size_lat", 9)).to(dev)
    enc.load_state_dict(ck["enc"]); enc.eval()
    if a.get("decoder") == "basedelta":
        dec = BaseDeltaDecoder(d_static=a["d_static"], static_grid=a["static_grid"],
                               d_dyn=a["d_dyn"], dyn_grid=a["dyn_grid"],
                               hidden_ch=a["dec_hidden_ch"]).to(dev)
    else:
        dec = SpatialGridDecoder(d_static=a["d_static"], static_grid=a["static_grid"],
                                 d_dyn=a["d_dyn"], hidden_ch=a["dec_hidden_ch"],
                                 chunk_size_lat=a.get("chunk_size_lat", 9),
                                 dyn_spatial=True, dyn_grid=a["dyn_grid"]).to(dev)
    dec.load_state_dict(ck["dec"]); dec.eval()
    return enc, dec


def probe_fit(X, Y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    ms = []
    for c in range(Y.shape[1]):
        if len(np.unique(Y[:, c])) < 2:
            ms.append(None); continue
        ms.append(LogisticRegression(max_iter=600).fit(sc.transform(X), Y[:, c]))
    return sc, ms


def probe_ap(sc, ms, X, Y):
    from sklearn.metrics import average_precision_score
    aps = []
    for c, m in enumerate(ms):
        if m is None or len(np.unique(Y[:, c])) < 2:
            continue
        aps.append(average_precision_score(Y[:, c], m.predict_proba(sc.transform(X))[:, 1]))
    return float(np.mean(aps)) if aps else float("nan")


@torch.no_grad()
def run(ck, loader, dev):
    enc, dec = build(ck, dev)
    REC, HYB, ATT, ATT_B, POS, POS_B = [], [], [], [], [], []
    for b in loader:
        seq = b["latents"].to(dev)
        B = seq.shape[0]
        if B < 2:
            continue
        g, z, _ = enc(seq)
        rec = dec(g[:, 0], z[:, 0])                       # normal reconstruction
        roll = torch.roll(torch.arange(B), 1)             # B = A rolled by one
        hyb = dec(g[:, 0], z[roll, 0])                    # zs_A + zd_B
        # summarise each latent the same way for both, so the probe sees like for like
        REC.append(rec.mean(2).flatten(1).cpu())
        HYB.append(hyb.mean(2).flatten(1).cpu())
        att = (b["attrs"] * b["slot_mask"][..., None]).amax(1)
        ATT.append((att > 0.5).float())
        ATT_B.append((att > 0.5).float()[roll])
        # motion signature: per-frame change of the decoded latent
        POS.append((rec[:, :, 1:] - rec[:, :, :-1]).abs().mean(2).flatten(1).cpu())
        POS_B.append((hyb[:, :, 1:] - hyb[:, :, :-1]).abs().mean(2).flatten(1).cpu())
    REC, HYB = torch.cat(REC).numpy(), torch.cat(HYB).numpy()
    ATT, ATT_B = torch.cat(ATT).numpy(), torch.cat(ATT_B).numpy()
    n = len(REC) // 2
    # fit identity probe on NORMAL reconstructions only
    sc, ms = probe_fit(REC[:n], ATT[:n])
    return {
        "id_from_recon": probe_ap(sc, ms, REC[n:], ATT[n:]),
        "id_from_swap": probe_ap(sc, ms, HYB[n:], ATT[n:]),       # matches A (zs donor)
        "id_leak_B": probe_ap(sc, ms, HYB[n:], ATT_B[n:]),        # matches B (zd donor)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--out", default="outputs/logs/swap.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    va = ClevrerSequence(args.cache_dir, 4, 400, "val", preload=True)
    dl = DataLoader(va, batch_size=32)
    print(f"[data] {len(va)} val videos\n")
    print(f"{'model':<22}{'id(recon)':>11}{'id(swap)=A':>12}{'id leak=B':>11}{'margin':>9}")
    print('-' * 65)
    res = {}
    for ck, lb in zip(args.ckpts, args.labels):
        r = run(ck, dl, dev); r["swap_margin"] = r["id_from_swap"] - r["id_leak_B"]
        res[lb] = r
        print(f"{lb:<22}{r['id_from_recon']:>11.3f}{r['id_from_swap']:>12.3f}"
              f"{r['id_leak_B']:>11.3f}{r['swap_margin']:>+9.3f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nmargin > 0 => the hybrid takes its identity from the z_static donor,"
          f"\ni.e. the split is real. margin ~ 0 => identity rides along with z_dyn.")
    print("SWAP_DONE")


if __name__ == "__main__":
    main()
