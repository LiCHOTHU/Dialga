"""Diagnostic GT-vs-model reconstruction for a v5.1.2 VAE-UNFROZEN checkpoint.

Renders 4 LABELED panels per frame so the eval validates itself:
  1. RAW                 - input frames (ground truth).
  2. Wan roundtrip (orig)- decode(encode_orig(pix)). CONTROL / VAE ceiling: this
                           uses the ORIGINAL pretrained VAE end-to-end, so it MUST
                           look good. If it doesn't, the render pipeline is buggy.
  3. e2 encoder roundtrip- decode(encode_ft(pix)). Isolates the fine-tuned
                           encoder's drift (no small model in the loop).
  4. e2 full model       - decode(small_dec(small_enc(encode_ft(pix)))).

The fine-tuned Wan VAE is loaded with strict=True (asserts an exact key match).
For an enc-unfrozen run the decoder is unchanged, so panels 2-4 share one decoder;
only the ENCODER differs between the original and fine-tuned VAEs.

Usage:
    python scripts/viz/save_v512_vae_recon.py --ckpt ... --cache_dir ... \\
        --video_dir ... --out_dir outputs/v512e2_recon --n_videos 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image, ImageDraw
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


def _vae_encode(vae, pix):
    x = pix.permute(0, 2, 1, 3, 4).contiguous().to(next(vae.parameters()).dtype)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.float()


def _vae_decode(vae, latent):
    z = latent.to(next(vae.parameters()).dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    return pix.permute(0, 2, 1, 3, 4).contiguous().float()


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 0.5 * 255.0
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()   # (T,H,W,3)


def label_panel(frames_u8, text):
    """frames_u8 (T,H,W,3) -> same, with `text` drawn top-left on each frame."""
    out = []
    for fr in frames_u8:
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, len(text) * 6 + 4, 12], fill=(0, 0, 0))
        d.text((2, 1), text, fill=(255, 255, 0))
        out.append(np.asarray(im))
    return np.stack(out, 0)


def load_vae(model_id, dtype, device, state_dict=None):
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    if state_dict is not None:
        vae.load_state_dict(state_dict, strict=True)   # asserts exact key match
        print("[vae] fine-tuned weights loaded with strict=True (exact match OK)")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--use_gt_gate", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    assert "wan_vae" in ckpt, "no fine-tuned wan_vae in ckpt"
    d_static = int(a.get("d_static", 64)); d_dyn = int(a.get("d_dyn", 64))
    d_state = int(a.get("d_state", 32));   d_event = int(a.get("d_event", 4))
    enc_hidden_ch = int(a.get("enc_hidden_ch", 128)); dec_hidden_ch = int(a.get("dec_hidden_ch", 256))
    chunk_size_lat = int(a.get("chunk_size_lat", 9)); no_proj = bool(a.get("no_proj", False))
    model_id = a.get("vae_model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    print(f"[ckpt] {args.ckpt} | ep={ckpt.get('epoch')} "
          f"enc_unfrozen={a.get('unfreeze_vae_enc')} dec_unfrozen={a.get('unfreeze_vae_dec')}")

    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch, use_layer_norm=use_ln).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_hidden_ch, chunk_size_lat=chunk_size_lat).to(device)
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    eh = EventHead(d_dyn=d_dyn, d_event=d_event).to(device)
    ge = GEvent(d_event=d_event, d_dyn=d_dyn).to(device)
    gp = GatePredictor(d_dyn=d_dyn).to(device)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    fwd.load_state_dict(ckpt["fwd"], strict=False)
    eh.load_state_dict(ckpt["event_head"]); ge.load_state_dict(ckpt["g_event"]); gp.load_state_dict(ckpt["gate_predictor"])
    for m in (enc, dec, fwd, eh, ge, gp): m.eval()

    print(f"[vae] loading ORIGINAL pretrained {model_id} (control)")
    vae_orig = load_vae(model_id, dtype, device, state_dict=None)
    print(f"[vae] loading FINE-TUNED VAE from ckpt")
    vae_ft = load_vae(model_id, dtype, device, state_dict=ckpt["wan_vae"])

    ds = ClevrerChunkPairsWithPixels(
        args.cache_dir, args.video_dir, image_size=128,
        seed=int(a.get("seed", 42)), max_videos=int(a.get("max_videos", 1000)),
        split="val", val_frac=float(a.get("val_frac", 0.2)))
    print(f"[data] val split: {len(ds)} pairs")

    @torch.no_grad()
    def roundtrip(vae, pix):
        return _vae_decode(vae, _vae_encode(vae, pix.unsqueeze(0).to(device))).squeeze(0).cpu()

    @torch.no_grad()
    def full_model(pix, gate_gt):
        lat = _vae_encode(vae_ft, pix.unsqueeze(0).to(device))
        out = enc(lat); z_static, z_dyn = out["z_static"], out["z_dyn"]
        rec = _vae_decode(vae_ft, dec(z_static, z_dyn)).squeeze(0).cpu()
        return rec

    seen, n = set(), 0
    for idx in range(len(ds)):
        s = ds[idx]; vid = int(s["video_id"])
        if vid in seen:
            continue
        seen.add(vid)
        pix = torch.cat([s["pix_obs"], s["pix_pred"]], dim=0)            # (66,3,H,W) raw

        p_raw  = to_uint8(pix)
        p_orig = to_uint8(torch.cat([roundtrip(vae_orig, s["pix_obs"]), roundtrip(vae_orig, s["pix_pred"])], 0))
        p_ften = to_uint8(torch.cat([roundtrip(vae_ft,   s["pix_obs"]), roundtrip(vae_ft,   s["pix_pred"])], 0))
        p_full = to_uint8(torch.cat([full_model(s["pix_obs"], 0.0),     full_model(s["pix_pred"], 0.0)], 0))

        T = min(len(p_raw), len(p_orig), len(p_ften), len(p_full))
        panels = [
            label_panel(p_raw[:T],  "RAW"),
            label_panel(p_orig[:T], "Wan rt (orig)"),
            label_panel(p_ften[:T], "e2 enc rt"),
            label_panel(p_full[:T], "e2 full"),
        ]
        sbs = np.concatenate(panels, axis=2)                            # along width
        out_path = out_dir / f"video_{vid:05d}_4panel.mp4"
        imageio.mimwrite(out_path, sbs, fps=8, codec="libx264", quality=8)
        print(f"  vid={vid:5d} frames={T} -> {out_path.name}")
        n += 1
        if n >= args.n_videos:
            break

    print(f"\n[done] wrote {n} videos to {out_dir}")
    print("Panels L->R: RAW | Wan roundtrip(orig=control) | e2 encoder roundtrip | e2 full model")


if __name__ == "__main__":
    main()
