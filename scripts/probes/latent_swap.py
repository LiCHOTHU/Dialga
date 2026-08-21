"""Latent-swap test: the decisive static/dynamic disentanglement check.

For clip pairs (A, B): encode both, decode the SWAPPED code
    x_swap = Decoder( z_static(A), z_dyn(B) )
then RE-ENCODE x_swap and ask whether the swap kept A's identity and B's motion:
    identity_follows_A : cos(z_static(swap), z_static(A)) > cos(., z_static(B))
    motion_follows_B   : cos(z_dyn(swap),    z_dyn(B))    > cos(., z_dyn(A))
A clean factorization -> both win-rates near 1.0. If ~0.5 the slots are not
truly separable (probe-deep only). A no-swap control (decode A's own codes,
re-encode) must recover A on BOTH slots -- it validates the decode->re-encode
loop so a failed swap can't be blamed on the loop.

Usage:
  python scripts/probes/latent_swap.py --ckpt .../v59_clevrer/v5_best.pt \\
      --cache_dir .../wan_10000vid_W33 --n_pairs 300 --out .../swap.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.ssv2_action_probe import build_encoder
from src.model.latent_decoder import SpatialGridDecoder


def build_decoder(ckpt, device):
    a = ckpt.get("args", {})
    g = lambda k, d: a[k] if k in a else d
    d_pose = int(g("d_pose", 0)) if g("use_camera_pose", False) else 0
    dec = SpatialGridDecoder(
        d_static=int(g("d_static", 96)), static_grid=int(g("static_grid", 4)),
        d_dyn=int(g("d_dyn", 256)), hidden_ch=int(g("dec_hidden_ch", 384)),
        chunk_size_lat=int(g("chunk_size_lat", 9)), depth=int(g("dec_depth", 2)),
        d_pose=d_pose, dyn_spatial=bool(g("dyn_spatial", True)),
        dyn_grid=int(g("dyn_grid", 8)),
    ).to(device)
    dec.load_state_dict(ckpt["decoder"]); dec.eval()
    return dec


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc, _ = build_encoder(ckpt, device); enc.load_state_dict(ckpt["encoder"]); enc.eval()
    dec = build_decoder(ckpt, device)
    print(f"[model] {args.ckpt}")

    meta = json.loads((Path(args.cache_dir) / "metadata.json").read_text())
    wins = meta["windows"]
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(wins))[: 2 * args.n_pairs]

    def load(i):
        b = torch.load(Path(args.cache_dir) / wins[i]["path"], map_location="cpu",
                       weights_only=False)
        return b["latent"].unsqueeze(0).to(device)

    def code(lat):
        o = enc(lat)
        return o["z_static_grid"], o["z_static"], o["z_dyn"]

    id_win, mo_win = [], []
    ctrl_id, ctrl_mo = [], []
    ids_A, ids_B, mos_A, mos_B = [], [], [], []
    for p in range(args.n_pairs):
        A, B = load(int(idx[2 * p])), load(int(idx[2 * p + 1]))
        gsA, zsA, zdA = code(A)
        gsB, zsB, zdB = code(B)
        # SWAP: static from A, dynamics from B
        x_swap = dec(gsA, zdB)
        _, zsS, zdS = code(x_swap)
        iA, iB = cos(zsS, zsA), cos(zsS, zsB)
        mB, mA = cos(zdS.mean(1), zdB.mean(1)), cos(zdS.mean(1), zdA.mean(1))
        id_win.append(iA > iB); mo_win.append(mB > mA)
        ids_A.append(iA); ids_B.append(iB); mos_A.append(mA); mos_B.append(mB)
        # CONTROL: no swap, decode A's own codes, must recover A on both
        x_ctrl = dec(gsA, zdA)
        _, zsC, zdC = code(x_ctrl)
        ctrl_id.append(cos(zsC, zsA) > cos(zsC, zsB))
        ctrl_mo.append(cos(zdC.mean(1), zdA.mean(1)) > cos(zdC.mean(1), zdB.mean(1)))

    res = {
        "ckpt": args.ckpt, "n_pairs": args.n_pairs,
        "swap": {
            "identity_follows_A_rate": round(float(np.mean(id_win)), 3),
            "motion_follows_B_rate": round(float(np.mean(mo_win)), 3),
            "cos_id_toA": round(float(np.mean(ids_A)), 3),
            "cos_id_toB": round(float(np.mean(ids_B)), 3),
            "cos_mo_toB": round(float(np.mean(mos_B)), 3),
            "cos_mo_toA": round(float(np.mean(mos_A)), 3),
        },
        "control_noswap": {
            "identity_recovers_A_rate": round(float(np.mean(ctrl_id)), 3),
            "motion_recovers_A_rate": round(float(np.mean(ctrl_mo)), 3),
        },
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    s = res["swap"]
    print(f"\n[verdict] identity_follows_A={s['identity_follows_A_rate']*100:.0f}%  "
          f"motion_follows_B={s['motion_follows_B_rate']*100:.0f}%  "
          f"(clean disentanglement -> both near 100%; chance=50%)")


if __name__ == "__main__":
    main()
