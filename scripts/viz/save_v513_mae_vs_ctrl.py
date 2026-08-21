"""Side-by-side reconstruction: v513 MAE smoke vs its lambda_mae=0 control.

Both checkpoints are FROZEN-VAE runs (no wan_vae key), trained on the same
20-video overfit subset (seed 42), differing only in --lambda_mae. Panels:

  1. RAW      - ground-truth frames.
  2. WAN-RT   - wan_decode(cached GT latent). VAE ceiling: best any model
                that decodes through frozen Wan can ever look.
  3. CTRL     - wan_decode(dec(enc(latent))) for the lambda_mae=0 ckpt.
  4. MAE      - same for the lambda_mae=0.5 ckpt.

Renders both train videos (seen during the overfit) and unseen videos, and
prints per-video PSNR for panels 2-4 so the visual diff has numbers attached.

Usage:
    python scripts/viz/save_v513_mae_vs_ctrl.py \
        --ckpt_mae .../v513_mae_smoke/last.pt \
        --ckpt_ctrl .../v513_mae_smoke_ctrl/last.pt \
        --cache_dir .../wan_10000vid_W33 --video_dir .../train_video \
        --out_dir outputs/viz/v513_mae_vs_ctrl --n_train 4 --n_unseen 2
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
from src.model.latent_decoder import LatentDecoder, SpatialBroadcastDecoder
from src.model.latent_encoder import LatentEncoder3D


def to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 0.5 * 255.0
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()   # (T,H,W,3)


def psnr(pred, gt):
    """pred/gt (T,3,H,W) in [-1,1] -> PSNR over [0,1] range."""
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


def load_small_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if not isinstance(a, dict):
        a = vars(a)
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
    else:
        dec = LatentDecoder(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                            hidden_ch=int(a["dec_hidden_ch"]),
                            chunk_size_lat=int(a.get("chunk_size_lat", 9))).to(device)
    enc.load_state_dict(ck["encoder"])
    dec.load_state_dict(ck["decoder"])
    enc.eval(); dec.eval()
    print(f"[ckpt] {ckpt_path} | ep={ck.get('epoch')} pool={pool_type} "
          f"dec={decoder_type} lambda_mae={a.get('lambda_mae')}")
    return enc, dec, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_mae", required=True)
    ap.add_argument("--ckpt_ctrl", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=4)
    ap.add_argument("--n_unseen", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vae_dtype", type=str, default="float32",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32}[args.vae_dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    enc_m, dec_m, a_m = load_small_model(args.ckpt_mae, device)
    enc_c, dec_c, a_c = load_small_model(args.ckpt_ctrl, device)
    assert float(a_m.get("lambda_mae", 0)) > 0 and float(a_c.get("lambda_mae", 0)) == 0, \
        "expected --ckpt_mae trained with lambda_mae>0 and --ckpt_ctrl with lambda_mae=0"

    model_id = a_m.get("vae_model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    print(f"[vae] loading frozen {model_id} ({args.vae_dtype})")
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=vdt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae = vae.to(device)

    @torch.no_grad()
    def wan_decode(latent):
        """latent (48,9,8,8) -> pixels (33,3,128,128) in [-1,1]."""
        z = latent.unsqueeze(0).to(device=device, dtype=vdt)
        out = vae.decode(z)
        pix = out.sample if hasattr(out, "sample") else out
        return pix.squeeze(0).permute(1, 0, 2, 3).contiguous().float().cpu()

    @torch.no_grad()
    def model_recon(enc, dec, latent):
        o = enc(latent.unsqueeze(0).to(device))
        return wan_decode(dec(o["z_static"], o["z_dyn"]).squeeze(0).cpu())

    # The 20 train videos: identical construction to the smoke run's dataset.
    ds_train = ClevrerChunkPairsWithPixels(
        args.cache_dir, args.video_dir, image_size=128, seed=42, max_videos=20)
    train_vids = sorted({int(ds_train.windows[p[0]]["video_id"]) for p in ds_train.pairs})
    print(f"[data] train subset: {len(ds_train)} pairs over videos {train_vids}")

    ds_all = ClevrerChunkPairsWithPixels(
        args.cache_dir, args.video_dir, image_size=128, seed=42, max_videos=0)

    def render(ds, want, skip_vids, tag):
        seen, rows = set(skip_vids), []
        for idx in range(len(ds)):
            vid = int(ds.windows[ds.pairs[idx][0]]["video_id"])
            if vid in seen:
                continue
            seen.add(vid)
            s = ds[idx]
            pix_gt = s["pix_obs"]                       # (33,3,128,128) raw
            p_raw = pix_gt
            p_wan = wan_decode(s["chunk_obs"])
            p_ctl = model_recon(enc_c, dec_c, s["chunk_obs"])
            p_mae = model_recon(enc_m, dec_m, s["chunk_obs"])
            T = min(len(p_raw), len(p_wan), len(p_ctl), len(p_mae))
            ps_wan, ps_ctl, ps_mae = (psnr(p[:T], p_raw[:T]) for p in (p_wan, p_ctl, p_mae))
            panels = [
                label_panel(to_uint8(p_raw[:T]), "RAW"),
                label_panel(to_uint8(p_wan[:T]), f"WAN-RT {ps_wan:.1f}dB"),
                label_panel(to_uint8(p_ctl[:T]), f"CTRL {ps_ctl:.1f}dB"),
                label_panel(to_uint8(p_mae[:T]), f"MAE {ps_mae:.1f}dB"),
            ]
            sbs = np.concatenate(panels, axis=2)
            out_path = out_dir / f"{tag}_video_{vid:05d}.mp4"
            imageio.mimwrite(out_path, sbs, fps=8, codec="libx264", quality=8)
            print(f"  [{tag}] vid={vid:5d} WAN={ps_wan:.2f} CTRL={ps_ctl:.2f} "
                  f"MAE={ps_mae:.2f} dB -> {out_path.name}")
            rows.append((tag, vid, ps_wan, ps_ctl, ps_mae))
            if len(rows) >= want:
                break
        return rows

    rows = render(ds_train, args.n_train, set(), "train")
    rows += render(ds_all, args.n_unseen, set(train_vids), "unseen")

    print("\nsummary (PSNR dB):")
    print(f"{'set':>7} {'vid':>6} {'WAN-RT':>7} {'CTRL':>7} {'MAE':>7} {'MAE-CTRL':>9}")
    for tag, vid, w, c, m in rows:
        print(f"{tag:>7} {vid:>6} {w:>7.2f} {c:>7.2f} {m:>7.2f} {m - c:>+9.2f}")
    print(f"\n[done] wrote {len(rows)} videos to {out_dir}")
    print("Panels L->R: RAW | Wan roundtrip (ceiling) | CTRL (no MAE) | MAE")


if __name__ == "__main__":
    main()
