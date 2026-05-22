"""Check 2: decoder magnitude sensitivity sweep.

Pick one val window, encode -> (z_static, z_dyn_enc). Hold z_static fixed,
scale z_dyn_enc to a set of target per-frame L2 norms (same directions,
different magnitudes), decode each through LatentDecoder -> Wan VAE -> pixels.
Save a horizontal strip of middle frames so blur/artifact onset is visible
as magnitude grows.

This isolates "does the decoder produce sharp output at the trained
magnitude and degrade smoothly as it extrapolates" from "are the
dynamics actually correct".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


def load_wan_vae(model_id, dtype, device):
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def wan_decode(vae, latent, device, dtype):
    z = latent.unsqueeze(0).to(device).to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    return pix.squeeze(0).permute(1, 0, 2, 3).contiguous().float().cpu()


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 0.5 * 255.0
    return x.to(torch.uint8).permute(1, 2, 0).contiguous()


def label_strip(width, text, font_h=14):
    """Tiny baseline-rendered label (just a colored bar; PIL kept optional)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (width, font_h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((2, 0), text, fill=(255, 255, 255))
        return torch.from_numpy(np.array(img))
    except Exception:
        return torch.zeros(font_h, width, 3, dtype=torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--vid_idx", type=int, default=0)
    ap.add_argument("--target_norms", type=str,
                    default="1,2,3,5,7,10,15,20")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--model_id", type=str,
                    default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "decoder_magnitude"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_norms = [float(x) for x in args.target_norms.split(",")]

    # --- load model ---
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn = int(a.get("d_dyn", 16))
    enc_hidden_ch = int(a.get("enc_hidden_ch", 32))
    dec_hidden_ch = int(a.get("dec_hidden_ch", 64))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_hidden_ch,
                        chunk_size_lat=chunk_size_lat).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    print(f"[model] loaded from {args.ckpt}")

    vae = load_wan_vae(args.model_id, dtype, device)

    # --- pick one val window ---
    cache_dir = Path(args.cache_dir)
    meta = json.loads((cache_dir / "metadata.json").read_text())
    windows = meta["windows"]
    w = windows[args.vid_idx]
    blob = torch.load(cache_dir / w["path"], map_location="cpu", weights_only=False)
    chunk_latent = blob["latent"]  # (C, T_lat, H, W)
    print(f"[data] vid={w['video_id']} start_frame={w['start_frame']} "
          f"latent_shape={tuple(chunk_latent.shape)}")

    # --- encode ---
    with torch.no_grad():
        out = enc(chunk_latent.unsqueeze(0).to(device))
        z_static = out["z_static"]       # (1, D_s)
        z_dyn = out["z_dyn"]              # (1, T, D_d)
    real_norms = z_dyn[0].norm(dim=-1)   # (T,)
    real_mean = real_norms.mean().item()
    print(f"[probe] z_dyn natural per-frame L2 norms: "
          f"min={real_norms.min().item():.2f} "
          f"mean={real_mean:.2f} "
          f"max={real_norms.max().item():.2f}")

    # unit-norm z_dyn (preserve direction) for clean scaling
    z_dyn_unit = z_dyn / (z_dyn.norm(dim=-1, keepdim=True) + 1e-8)

    # --- sweep magnitudes ---
    panels = []
    norms_with_real = sorted(set(target_norms + [round(real_mean, 2)]))
    print(f"[sweep] norms: {norms_with_real}  (real anchor={real_mean:.2f})")
    for n in norms_with_real:
        z_dyn_scaled = z_dyn_unit * n
        with torch.no_grad():
            recon_latent = dec(z_static, z_dyn_scaled).squeeze(0)
            pix = wan_decode(vae, recon_latent, device, dtype)
        mid = pix.shape[0] // 2
        frame = to_uint8(pix[mid])
        tag = f"||z||={n:.1f}" + ("  <-real" if abs(n - real_mean) < 1e-3 else "")
        lbl = label_strip(frame.shape[1], tag)
        panels.append(torch.cat([lbl, frame], dim=0))
        print(f"  norm={n:5.2f}  middle-frame stats: "
              f"mean={pix[mid].mean().item():+.3f} std={pix[mid].std().item():.3f} "
              f"min={pix[mid].min().item():+.3f} max={pix[mid].max().item():+.3f}")

    # also include GT-decoded middle frame for visual anchor
    with torch.no_grad():
        gt_pix = wan_decode(vae, chunk_latent, device, dtype)
    gt_frame = to_uint8(gt_pix[gt_pix.shape[0] // 2])
    gt_lbl = label_strip(gt_frame.shape[1], "GT")
    panels.insert(0, torch.cat([gt_lbl, gt_frame], dim=0))

    strip = torch.cat(panels, dim=1).numpy()   # (H, W_total, 3)
    ckpt_tag = Path(args.ckpt).stem  # v5 or v5_best
    out_path = out_dir / f"sweep_{ckpt_tag}_vid{args.vid_idx}.png"
    imageio.imwrite(out_path, strip)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
