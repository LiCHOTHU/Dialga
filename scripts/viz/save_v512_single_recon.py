"""Single-model reconstruction render for a FROZEN-VAE v5.1.2 run (e.g. v512_clevrer_big).

These runs read cached Wan latents (no wan_vae in ckpt) and decode through the
original frozen Wan VAE. Three labeled panels per frame, on the VAL split:

  1. RAW       - ground-truth frames.
  2. WAN-RT    - wan_decode(cached GT latent). The VAE ceiling: the best any model
                 decoding through frozen Wan can look.
  3. MODEL     - wan_decode(dec(enc(cached latent))). Our small enc+dec.

obs + pred chunks are concatenated so motion is visible across the full window.
Per-video PSNR (vs RAW) printed for panels 2-3.

Usage:
    python scripts/viz/save_v512_single_recon.py \
        --ckpt .../v512_clevrer_big/last.pt \
        --cache_dir .../wan_10000vid_W33 --video_dir .../train_video \
        --out_dir outputs/viz/v512_big_recon_ep165 --n_videos 6
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image, ImageDraw
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.model.latent_decoder import LatentDecoder, SpatialBroadcastDecoder, SlotDecoder
from src.model.latent_encoder import LatentEncoder3D


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 0.5 * 255.0
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()   # (T,H,W,3)


def psnr(pred, gt):
    mse = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def label_panel(frames_u8, text):
    out = []
    for fr in frames_u8:
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, len(text) * 6 + 4, 12], fill=(0, 0, 0))
        d.text((2, 1), text, fill=(255, 255, 0))
        out.append(np.asarray(im))
    return np.stack(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=6)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vae_dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32}[args.vae_dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    if not isinstance(a, dict):
        a = vars(a)
    assert "wan_vae" not in ck, "this is a VAE-UNFROZEN ckpt; use save_v512_vae_recon.py"
    use_ln = "norm_static.weight" in ck["encoder"]
    pool_type = a.get("pool_type", "mean")
    decoder_type = a.get("decoder_type", "linear")
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=pool_type,
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
    else:
        dec = LatentDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                            hidden_ch=int(a["dec_hidden_ch"]),
                            chunk_size_lat=int(a.get("chunk_size_lat", 9))).to(device)
    enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
    enc.eval(); dec.eval()
    print(f"[ckpt] {args.ckpt} | ep={ck.get('epoch')} pool={pool_type} dec={decoder_type}")

    model_id = a.get("vae_model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    print(f"[vae] loading frozen {model_id} ({args.vae_dtype})")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=vdt)
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

    @torch.no_grad()
    def model_recon(latent):
        o = enc(latent.unsqueeze(0).to(device))
        # SlotDecoder consumes the per-slot code z_slots (B, M, D_s); all other
        # decoders consume the pooled global z_static. Matches train_v5.py.
        cond = o["z_slots"] if isinstance(dec, SlotDecoder) else o["z_static"]
        return wan_decode(dec(cond, o["z_dyn"]).squeeze(0).cpu())

    ds = ClevrerChunkPairsWithPixels(
        args.cache_dir, args.video_dir, image_size=128,
        seed=int(a.get("seed", 42)), max_videos=int(a.get("max_videos", 0)),
        split=args.split, val_frac=float(a.get("val_frac", 0.2)))
    print(f"[data] {args.split} split: {len(ds)} pairs")

    seen, rows, n = set(), [], 0
    for idx in range(len(ds)):
        s = ds[idx]
        vid = int(s.get("video_id", ds.windows[ds.pairs[idx][0]]["video_id"]))
        if vid in seen:
            continue
        seen.add(vid)

        # observed chunk (+ predicted chunk if present) concatenated along time
        lat_obs = s["chunk_obs"]; pix_obs = s["pix_obs"]
        lats = [lat_obs]; pixs = [pix_obs]
        if "chunk_pred" in s and "pix_pred" in s:
            lats.append(s["chunk_pred"]); pixs.append(s["pix_pred"])

        p_raw = torch.cat(pixs, 0)
        p_wan = torch.cat([wan_decode(l) for l in lats], 0)
        p_mdl = torch.cat([model_recon(l) for l in lats], 0)
        T = min(len(p_raw), len(p_wan), len(p_mdl))
        ps_wan, ps_mdl = psnr(p_wan[:T], p_raw[:T]), psnr(p_mdl[:T], p_raw[:T])

        panels = [
            label_panel(to_uint8(p_raw[:T]), "RAW"),
            label_panel(to_uint8(p_wan[:T]), f"WAN-RT {ps_wan:.1f}dB"),
            label_panel(to_uint8(p_mdl[:T]), f"MODEL {ps_mdl:.1f}dB"),
        ]
        sbs = np.concatenate(panels, axis=2)
        out_path = out_dir / f"val_video_{vid:05d}.mp4"
        imageio.mimwrite(out_path, sbs, fps=8, codec="libx264", quality=8)
        print(f"  vid={vid:5d} frames={T} WAN={ps_wan:.2f} MODEL={ps_mdl:.2f} dB "
              f"(gap {ps_wan - ps_mdl:.2f}) -> {out_path.name}")
        rows.append((vid, ps_wan, ps_mdl))
        n += 1
        if n >= args.n_videos:
            break

    print("\nsummary (PSNR dB, val split):")
    print(f"{'vid':>6} {'WAN-RT':>7} {'MODEL':>7} {'gap':>7}")
    for vid, w, m in rows:
        print(f"{vid:>6} {w:>7.2f} {m:>7.2f} {w - m:>7.2f}")
    if rows:
        ws = [r[1] for r in rows]; ms = [r[2] for r in rows]
        print(f"{'MEAN':>6} {sum(ws)/len(ws):>7.2f} {sum(ms)/len(ms):>7.2f} "
              f"{sum(ws)/len(ws) - sum(ms)/len(ms):>7.2f}")
    print(f"\n[done] wrote {n} videos to {out_dir}")
    print("Panels L->R: RAW | Wan roundtrip (ceiling) | MODEL (our enc+dec)")


if __name__ == "__main__":
    main()
