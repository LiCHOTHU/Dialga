"""Pixel PSNR of the local models against the frozen Wan-VAE ceiling.

Everything measured tonight has been LATENT MSE, which cannot say how close we are to
the VAE. Two different questions get answered here, and they are not the same number:

  ceiling_psnr   PSNR(wan_decode(cached latent), ground-truth pixels)
                 what the frozen VAE alone achieves -- the upper bound, since our code
                 is distilled from that latent and cannot beat it
  model_psnr     PSNR(wan_decode(our reconstruction), ground-truth pixels)
                 what we achieve. ceiling - model = the price of the compression
  fidelity_psnr  PSNR(our pixels, the VAE's own pixels)
                 how faithfully we reproduce the VAE's output, independent of how good
                 the VAE was on this clip

Ground-truth frames come from the source mp4s via ClevrerPairedDataset, rebuilt with
the wan cache's own args so window indices line up with the cached latents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cache_wan_latents import load_wan_vae                    # noqa: E402
from src.data.clevrer_paired import ClevrerPairedDataset              # noqa: E402
from src.model.base_delta_decoder import BaseDeltaDecoder             # noqa: E402
from src.model.latent_decoder import SpatialGridDecoder               # noqa: E402
from src.model.memory_encoder import MemoryEncoder                    # noqa: E402


@torch.no_grad()
def wan_decode(vae, latent, device, dtype):
    z = latent.to(device).to(dtype)
    out = vae.decode(z)
    x = out.sample if hasattr(out, "sample") else out
    return x.float().clamp(-1, 1)                       # (B,3,T,H,W) in [-1,1]


def psnr(a, b):
    """a,b in [-1,1] -> dB, per clip then averaged."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return (10 * torch.log10(4.0 / mse))                # peak-to-peak = 2 -> 2^2 = 4


def build(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    return enc, dec, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--cache_dir", default="outputs/cache/clevrer_W33_10k")
    ap.add_argument("--n_chunks", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="outputs/logs/psnr.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    dtype = torch.float16
    meta = json.loads((Path(args.cache_dir) / "metadata.json").read_text())
    wargs, windows = meta["args"], meta["windows"]

    ds = ClevrerPairedDataset(
        data_dir=wargs["data_dir"], annotation_dir=wargs["annotation_dir"],
        split=wargs["split"], window_length=wargs["window_length"],
        frames_per_video=wargs["frames_per_video"],
        windows_per_video=wargs["windows_per_video"],
        max_videos=wargs["max_videos"], max_objects=wargs["max_objects"],
        coordinate_mode="world_xy", image_size=wargs["image_size"], seed=wargs["seed"],
        deterministic_starts=[int(s) for s in wargs["deterministic_starts"].split(",")])

    # val videos are every-10th in ClevrerSequence's split; take windows from the tail
    idxs = list(range(len(windows) - args.n_chunks, len(windows)))
    print(f"[data] {len(idxs)} held-out chunks", flush=True)

    vae = load_wan_vae("Wan-AI/Wan2.2-TI2V-5B-Diffusers", dtype, dev)
    labels = args.labels or [Path(c).parent.name for c in args.ckpts]
    res = {}

    for ck, lbl in zip(args.ckpts, labels):
        enc, dec, a = build(ck, dev)
        C, M, F = [], [], []
        for s in range(0, len(idxs), args.batch):
            ids = idxs[s:s + args.batch]
            lat = torch.stack([torch.load(Path(args.cache_dir) / windows[i]["path"],
                                          map_location="cpu",
                                          weights_only=False)["latent"].float()
                               for i in ids]).to(dev)                 # (B,C,T,H,W)
            gt = torch.stack([ds[i]["frames"] for i in ids]).to(dev)  # (B,W,3,H,W)
            gt = gt.permute(0, 2, 1, 3, 4)                            # (B,3,W,H,W)
            g, z, _ = enc(lat.unsqueeze(1))                           # one chunk
            rec = dec(g[:, 0], z[:, 0])
            ceil_pix = wan_decode(vae, lat, dev, dtype)
            mdl_pix = wan_decode(vae, rec, dev, dtype)
            T = min(gt.shape[2], ceil_pix.shape[2], mdl_pix.shape[2])
            gt, ceil_pix, mdl_pix = gt[:, :, :T], ceil_pix[:, :, :T], mdl_pix[:, :, :T]
            C.append(psnr(ceil_pix, gt).cpu())
            M.append(psnr(mdl_pix, gt).cpu())
            F.append(psnr(mdl_pix, ceil_pix).cpu())
        c, m, f = (torch.cat(x).mean().item() for x in (C, M, F))
        res[lbl] = {"ceiling_psnr": c, "model_psnr": m, "fidelity_psnr": f,
                    "gap_db": c - m}
        print(f"{lbl:<28} ceiling {c:6.2f} dB | model {m:6.2f} dB | "
              f"gap {c - m:5.2f} dB | fidelity-to-VAE {f:6.2f} dB", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[saved] {args.out}\nPSNR_DONE")


if __name__ == "__main__":
    main()
