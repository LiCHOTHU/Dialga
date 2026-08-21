"""Compute a comparable PSNR table across ALL CLEVRER runs.

Every v5.1.2+ CLEVRER run shares the same val split (seed=42, val_frac=0.2,
same cache_dir), so the Wan-VAE roundtrip ceiling and the val clips are
identical across models — a fair head-to-head. This script:

  1. loads the frozen Wan VAE once,
  2. picks the first N distinct val videos,
  3. computes the WAN-RT ceiling once (model-independent),
  4. for each checkpoint, builds its enc+dec (respecting pool_type/decoder_type)
     and computes MODEL PSNR on the same clips.

Prints a single table (PSNR dB vs RAW, mean over the N clips) and the gap.
No mp4 written — numbers only, so it's fast.
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.model.latent_decoder import (LatentDecoder, SpatialBroadcastDecoder, SlotDecoder,
                                       SpatialGridDecoder)
from src.model.latent_encoder import LatentEncoder3D


def psnr(pred, gt):
    mse = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def build_enc_dec(a, enc_state, dec_state, device):
    use_ln = "norm_static.weight" in enc_state
    pool_type = a.get("pool_type", "mean")
    decoder_type = a.get("decoder_type", "linear")
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=pool_type,
                          static_grid=(int(a["static_grid"]) if a.get("static_grid") else None),
                          n_queries=int(a.get("pool_queries", 8)),
                          n_heads=int(a.get("pool_heads", 4))).to(device)
    if decoder_type == "broadcast":
        dec = SpatialBroadcastDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                                      hidden_ch=int(a["dec_hidden_ch"]),
                                      chunk_size_lat=int(a.get("chunk_size_lat", 9)),
                                      depth=int(a.get("dec_depth", 3))).to(device)
    elif decoder_type == "slot":
        dec = SlotDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["dec_hidden_ch"]),
                          chunk_size_lat=int(a.get("chunk_size_lat", 9)),
                          depth=int(a.get("dec_depth", 3))).to(device)
    elif decoder_type == "spatial":
        dec = SpatialGridDecoder(d_static=int(a["d_static"]),
                                 static_grid=int(a.get("static_grid", 4)),
                                 d_dyn=int(a["d_dyn"]),
                                 hidden_ch=int(a["dec_hidden_ch"]),
                                 chunk_size_lat=int(a.get("chunk_size_lat", 9)),
                                 depth=int(a.get("dec_depth", 3))).to(device)
    else:
        dec = LatentDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                            hidden_ch=int(a["dec_hidden_ch"]),
                            chunk_size_lat=int(a.get("chunk_size_lat", 9))).to(device)
    enc.load_state_dict(enc_state); dec.load_state_dict(dec_state)
    enc.eval(); dec.eval()
    return enc, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="label=path pairs, e.g. big=/.../v5_best.pt")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vae_dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32}[args.vae_dtype]

    print(f"[vae] loading frozen Wan VAE ({args.vae_dtype})")
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                           subfolder="vae", torch_dtype=vdt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae = vae.to(device)

    @torch.no_grad()
    def wan_decode(latent):
        z = latent.unsqueeze(0).to(device=device, dtype=vdt)
        out = vae.decode(z)
        pix = out.sample if hasattr(out, "sample") else out
        return pix.squeeze(0).permute(1, 0, 2, 3).contiguous().float().cpu()

    # shared val clips (identical split across all runs)
    ds = ClevrerChunkPairsWithPixels(args.cache_dir, args.video_dir, image_size=128,
                                     seed=42, max_videos=0, split="val", val_frac=0.2)
    seen, clips = set(), []
    for idx in range(len(ds)):
        s = ds[idx]
        vid = int(s.get("video_id", ds.windows[ds.pairs[idx][0]]["video_id"]))
        if vid in seen:
            continue
        seen.add(vid)
        lats = [s["chunk_obs"]]; pixs = [s["pix_obs"]]
        if "chunk_pred" in s and "pix_pred" in s:
            lats.append(s["chunk_pred"]); pixs.append(s["pix_pred"])
        clips.append((vid, lats, pixs))
        if len(clips) >= args.n_videos:
            break
    print(f"[data] {len(clips)} distinct val clips")

    # Wan-VAE ceiling (model independent) + cache RAW/WAN-RT pixels per clip
    ceil_ps, raw_cache, wan_cache = [], [], []
    for vid, lats, pixs in clips:
        p_raw = torch.cat(pixs, 0)
        p_wan = torch.cat([wan_decode(l) for l in lats], 0)
        T = min(len(p_raw), len(p_wan))
        raw_cache.append(p_raw[:T]); wan_cache.append(p_wan[:T])
        ceil_ps.append(psnr(p_wan[:T], p_raw[:T]))
    ceiling = sum(ceil_ps) / len(ceil_ps)
    print(f"[ceiling] Wan-VAE roundtrip mean PSNR = {ceiling:.2f} dB\n")

    rows = []
    for spec in args.ckpts:
        label, path = spec.split("=", 1)
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
            enc, dec = build_enc_dec(a, ck["encoder"], ck["decoder"], device)
            mps = []
            with torch.no_grad():
                for i, (vid, lats, pixs) in enumerate(clips):
                    outs = []
                    for l in lats:
                        o = enc(l.unsqueeze(0).to(device))
                        if isinstance(dec, SlotDecoder):
                            cond = o["z_slots"]
                        elif isinstance(dec, SpatialGridDecoder):
                            cond = o["z_static_grid"]
                        else:
                            cond = o["z_static"]
                        outs.append(wan_decode(dec(cond, o["z_dyn"]).squeeze(0).cpu()))
                    p_mdl = torch.cat(outs, 0)[:len(raw_cache[i])]
                    mps.append(psnr(p_mdl, raw_cache[i]))
            m = sum(mps) / len(mps)
            rows.append((label, ck.get("epoch"), a.get("pool_type", "mean"),
                         a.get("decoder_type", "linear"), m))
            print(f"  {label:<24} ep{str(ck.get('epoch')):<5} MODEL={m:6.2f} dB  gap={ceiling-m:5.2f}")
        except Exception as e:
            print(f"  {label:<24} FAILED: {e}")

    print(f"\n{'='*72}\nPSNR table (dB, mean over {len(clips)} val clips)\n{'='*72}")
    print(f"{'run':<24}{'ep':>5}{'pool':>7}{'decoder':>10}{'PSNR':>8}{'gap':>7}")
    print("-" * 72)
    print(f"{'Wan-VAE ceiling':<24}{'':>5}{'':>7}{'':>10}{ceiling:>8.2f}{0.0:>7.2f}")
    for label, ep, pool, dec_t, m in sorted(rows, key=lambda r: -r[4]):
        print(f"{label:<24}{str(ep):>5}{pool:>7}{dec_t:>10}{m:>8.2f}{ceiling-m:>7.2f}")


if __name__ == "__main__":
    main()
