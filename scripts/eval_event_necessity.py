"""scripts/eval_event_necessity.py — TODO 3.

Does z_event provide a real correction at collision chunks?

For each val chunk pair, compute L_pred (latent MSE of decoded prediction vs
chunk_pred latent) in two settings:

    (a) with event injection:   z_dyn_pred = z_dyn_base + g_event(eh(z_dyn_last))
    (b) without event injection: z_dyn_pred = z_dyn_base

Bucket by gate_GT (1 = collision in chunk_pred, 0 = no collision). Expected:

    | bucket            | L_pred (a) with | L_pred (b) without | gain |
    | gate_GT=0         | ≈                |  ≈                  |  ≈ 0 |   (sanity)
    | gate_GT=1         | smaller          |  larger             |  > 0 |   (events help)

If the "events help" gap is meaningfully positive, the event channel does work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="v5_events.pt from train_v5_events.py")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn    = int(a.get("d_dyn", 16))
    d_state  = int(a.get("d_state", 8))
    d_event  = int(a.get("d_event", 4))
    enc_hid  = int(a.get("enc_hidden_ch", 32))
    dec_hid  = int(a.get("dec_hidden_ch", 64))
    chunk_T  = int(a.get("chunk_size_lat", 9))
    no_proj  = bool(a.get("no_proj", False))
    shared_trunk = bool(a.get("shared_trunk", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn,
                          hidden_ch=enc_hid, shared_trunk=shared_trunk).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn,
                        hidden_ch=dec_hid, chunk_size_lat=chunk_T).to(device)
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    eh = EventHead(d_dyn=d_dyn, d_event=d_event).to(device)
    ge = GEvent(d_event=d_event, d_dyn=d_dyn).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    fwd.load_state_dict(ckpt["fwd"])
    eh.load_state_dict(ckpt["event_head"])
    ge.load_state_dict(ckpt["g_event"])
    for m in (enc, dec, fwd, eh, ge): m.eval()
    print(f"[model] loaded v5_events from {args.ckpt}")

    ds = ClevrerChunkPairs(args.cache_dir, split="val",
                            val_frac=args.val_frac, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=chunk_collate)

    sums = {"event_inject_event": 0.0, "event_inject_nonevent": 0.0,
            "no_inject_event":    0.0, "no_inject_nonevent":    0.0}
    cnt =  {"event_inject_event": 0,   "event_inject_nonevent": 0,
            "no_inject_event":    0,   "no_inject_nonevent":    0}

    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            chunk_obs  = batch["chunk_obs"].to(device)
            chunk_pred = batch["chunk_pred"].to(device)
            gate_GT    = batch["gate_GT"].to(device).float()
            enc_obs = enc(chunk_obs)
            z_static = enc_obs["z_static"]
            z_dyn_obs = enc_obs["z_dyn"]
            z_dyn_last = z_dyn_obs[:, -1]
            T = z_dyn_obs.shape[1]

            z_dyn_base = fwd.chunk_step(z_dyn_last, T)
            correction = ge(eh(z_dyn_last)).unsqueeze(1).expand(-1, T, -1)

            # (a) event injection (gate=1 chunks get the residual added)
            z_dyn_a = z_dyn_base + correction * gate_GT.unsqueeze(-1).unsqueeze(-1)
            # (b) no injection at all
            z_dyn_b = z_dyn_base

            wan_a = dec(z_static, z_dyn_a)
            wan_b = dec(z_static, z_dyn_b)

            err_a = ((wan_a - chunk_pred) ** 2).mean(dim=(1, 2, 3, 4))      # (B,)
            err_b = ((wan_b - chunk_pred) ** 2).mean(dim=(1, 2, 3, 4))

            for i in range(gate_GT.shape[0]):
                is_event = bool(gate_GT[i].item() > 0.5)
                if is_event:
                    sums["event_inject_event"]    += float(err_a[i]); cnt["event_inject_event"] += 1
                    sums["no_inject_event"]       += float(err_b[i]); cnt["no_inject_event"]    += 1
                else:
                    sums["event_inject_nonevent"] += float(err_a[i]); cnt["event_inject_nonevent"] += 1
                    sums["no_inject_nonevent"]    += float(err_b[i]); cnt["no_inject_nonevent"]    += 1

    means = {k: sums[k] / max(cnt[k], 1) for k in sums}
    print(f"[encode] {sum(cnt.values())} chunks in {time.time()-t0:.1f}s")
    print()
    print("                       L_pred (latent MSE)")
    print(f"                   |  gate_GT=1 (event)  |  gate_GT=0 (non-event)  |  n_chunks")
    print(f"  with injection   |  {means['event_inject_event']:.5f}             |  {means['event_inject_nonevent']:.5f}                |  {cnt['event_inject_event']} / {cnt['event_inject_nonevent']}")
    print(f"  no injection     |  {means['no_inject_event']:.5f}             |  {means['no_inject_nonevent']:.5f}                |  {cnt['no_inject_event']} / {cnt['no_inject_nonevent']}")
    print()
    gain_event    = means["no_inject_event"]    - means["event_inject_event"]
    gain_nonevent = means["no_inject_nonevent"] - means["event_inject_nonevent"]
    print(f"Δ L_pred (no_inject − inject) on EVENT chunks    = {gain_event:+.5f}    ← positive means events HELP")
    print(f"Δ L_pred (no_inject − inject) on NONEVENT chunks = {gain_nonevent:+.5f}    ← should be ≈ 0 (sanity)")

    out_path = Path(args.ckpt).parent / "event_necessity.json"
    out_path.write_text(json.dumps({"means": means, "counts": cnt,
                                    "gain_event": gain_event,
                                    "gain_nonevent": gain_nonevent,
                                    "args": vars(args)}, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
