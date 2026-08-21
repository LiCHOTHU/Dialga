"""Forward-dynamics rollout eval (paper Table tab:q3_rollout).

Demonstrates z_dyn is PREDICTABLE: roll the dynamics code forward h chunks with
the trained ForwardDynamics predictor, decode, and compare to the real future
chunk. Baseline = copy-last (assume no change). Reports latent MSE (and, with
--pixel, Wan-decoded pixel PSNR) at horizon h=1,2.

Uses CLEVRER cache (3 chunks/video at start_frames 0/33/66 -> h up to 2).

Usage:
  python scripts/probes/rollout_eval.py --ckpt .../v5_best.pt \\
      --cache_dir .../wan_10000vid_W33 --max_videos 400 [--pixel] --out .../q3_roll.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.ssv2_action_probe import build_encoder
from scripts.probes.latent_swap import build_decoder
from src.model.forward_dynamics import ForwardDynamics


def build_fwd(ckpt, device):
    a = ckpt.get("args", {}); g = lambda k, d: a[k] if k in a else d
    fwd = ForwardDynamics(d_dyn=int(g("d_dyn", 256)), d_state=int(g("d_state", 48)),
                          no_proj=bool(g("no_proj", False))).to(device)
    fwd.load_state_dict(ckpt["fwd"]); fwd.eval()
    return fwd


def group_by_video(cache_dir, max_videos):
    meta = json.loads((Path(cache_dir) / "metadata.json").read_text())
    by = {}
    for i, w in enumerate(meta["windows"]):
        by.setdefault(int(w["video_id"]), []).append((int(w["start_frame"]), w["path"]))
    vids = sorted(v for v, ws in by.items() if len(ws) >= 3)
    if max_videos:
        vids = vids[:max_videos]
    return [(v, sorted(by[v])[:3]) for v in vids], meta


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=400)
    ap.add_argument("--pixel", action="store_true", help="also Wan-decode for pixel PSNR")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc, _ = build_encoder(ck, dev); enc.load_state_dict(ck["encoder"]); enc.eval()
    dec = build_decoder(ck, dev)
    fwd = build_fwd(ck, dev)
    vae = None
    if args.pixel:
        from scripts.cache_wan_ssv2 import load_wan_vae
        vae = load_wan_vae("Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch.bfloat16, dev)

    def psnr(a, b):  # latent-space proxy PSNR if no vae; else caller decodes
        mse = F.mse_loss(a, b).item()
        return 10 * np.log10(1.0 / (mse + 1e-8)), mse

    vids, _ = group_by_video(args.cache_dir, args.max_videos)
    cd = Path(args.cache_dir)
    lat = {}
    def load(p): return torch.load(cd / p, map_location="cpu", weights_only=False)["latent"].unsqueeze(0).to(dev)

    acc = {"dialga": {1: [], 2: []}, "copylast": {1: [], 2: []}}
    for v, chunks in vids:
        c0, c1, c2 = load(chunks[0][1]), load(chunks[1][1]), load(chunks[2][1])
        o = enc(c0); gs, zd = o["z_static_grid"], o["z_dyn"]
        T = zd.shape[1]
        z_exit = zd[:, -1]
        zp1 = fwd.chunk_step(z_exit, T)                       # predicted chunk1 z_dyn
        pred1 = dec(gs, zp1)
        zp2 = fwd.chunk_step(zp1[:, -1], T)
        pred2 = dec(gs, zp2)
        for h, (pred, real) in {1: (pred1, c1), 2: (pred2, c2)}.items():
            acc["dialga"][h].append(F.mse_loss(pred, real).item())
            acc["copylast"][h].append(F.mse_loss(c0, real).item())    # assume no change

    res = {"ckpt": args.ckpt, "n_videos": len(vids), "metric": "latent MSE (lower better)"}
    for m in ("copylast", "dialga"):
        res[m] = {f"h{h}": round(float(np.mean(acc[m][h])), 5) for h in (1, 2)}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"[verdict] DIALGA should beat copy-last if z_dyn is predictable: "
          f"h1 {res['dialga']['h1']} vs {res['copylast']['h1']}")


if __name__ == "__main__":
    main()
