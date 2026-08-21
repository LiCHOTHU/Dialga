"""Render the object-centric (mask-weighted) reconstruction of an overfit clip.

Loads an oc_overfit checkpoint (frozen encoder + object-centric fine-tuned
decoder), encodes one CLEVRER chunk, decodes through the frozen Wan VAE to
pixels, and saves a side-by-side video:
    [ GT | reconstruction | reconstruction x object-mask (objects only) ]
so the object sharpening from the mask-weighted loss is directly visible.
Prints object-region PSNR and whole-frame PSNR for the rendered clip.
"""
import argparse, math, sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from torch.utils.data import DataLoader
from diffusers import AutoencoderKLWan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.data.clevrer_window import chunk_collate
from src.model.latent_decoder import (LatentDecoder, SpatialBroadcastDecoder,
                                       SlotDecoder, SpatialGridDecoder)
from src.model.latent_encoder import LatentEncoder3D


def build_enc(a, state, dev):
    use_ln = "norm_static.weight" in state
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=a.get("pool_type", "mean"),
                          n_queries=int(a.get("pool_queries", 8)),
                          n_heads=int(a.get("pool_heads", 4)),
                          static_grid=int(a.get("static_grid", 4))).to(dev)
    enc.load_state_dict(state); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_dec(a, dev):
    dt = a.get("decoder_type", "linear")
    kw = dict(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
              hidden_ch=int(a["dec_hidden_ch"]), chunk_size_lat=int(a.get("chunk_size_lat", 9)))
    if dt == "broadcast":
        return SpatialBroadcastDecoder(depth=int(a.get("dec_depth", 3)), **kw).to(dev)
    if dt == "slot":
        return SlotDecoder(depth=int(a.get("dec_depth", 3)), **kw).to(dev)
    if dt == "spatial":
        return SpatialGridDecoder(static_grid=int(a.get("static_grid", 4)),
                                  depth=int(a.get("dec_depth", 3)), **kw).to(dev)
    return LatentDecoder(**kw).to(dev)


def to_u8(x):  # (T,3,H,W) in [-1,1] -> (T,H,W,3) uint8
    return (((x.clamp(-1, 1) + 1) / 2) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def psnr(pred, gt, w):
    err = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean(dim=1, keepdim=True)
    mse = (err * w).sum() / w.sum().clamp(min=1.0)
    return 10.0 * math.log10(1.0 / max(mse.item(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video_dir", default="/storage/project/r-agarg35-0/lwang831/"
                    "dataset/CLEVRER/train_video")
    ap.add_argument("--out", default="outputs/oc_overfit_recon.mp4")
    ap.add_argument("--sample", type=int, default=0, help="which clip")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae_dtype", default="bfloat16")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.vae_dtype]
    if dev.type == "cpu":
        vdt = torch.float32

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
    enc = build_enc(a, ck["encoder"], dev)
    dec = build_dec(a, dev); dec.load_state_dict(ck["decoder"]); dec.eval()
    is_slot = isinstance(dec, SlotDecoder)
    print(f"[ckpt] {args.ckpt} ep={ck.get('epoch')} bg_weight={ck.get('bg_weight')} "
          f"obj_psnr(train)={ck.get('obj_psnr')}")

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                           subfolder="vae", torch_dtype=vdt).eval().to(dev)
    for p in vae.parameters():
        p.requires_grad_(False)

    ds = ClevrerChunkPairsWithPixels(cache_dir=a["cache_dir"], video_dir=args.video_dir,
                                     image_size=128, split=args.split,
                                     val_frac=float(a.get("val_frac", 0.2)),
                                     seed=int(a.get("seed", 42)), max_videos=args.max_videos,
                                     return_masks=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=chunk_collate)
    b = list(dl)[args.sample % len(ds)]

    with torch.no_grad():
        x = b["chunk_obs"].to(dev); z = enc(x)
        if is_slot:
            cond = z["z_slots"]
        elif isinstance(dec, SpatialGridDecoder):
            cond = z["z_static_grid"]
        else:
            cond = z["z_static"]
        lat = dec(cond, z["z_dyn"])
        out = vae.decode(lat.to(vdt))
        pix = (out.sample if hasattr(out, "sample") else out).permute(0, 2, 1, 3, 4).float()
        # Wan-VAE ceiling: decode the cached GT latent directly (no DIALGA enc/dec)
        vout = vae.decode(x.to(vdt))
        vae_pix = (vout.sample if hasattr(vout, "sample") else vout).permute(0, 2, 1, 3, 4).float()
    gt = b["pix_obs"].to(dev); m = b["mask_obs"].to(dev)              # (1,T,3/1,H,W)
    print(f"[recon] obj-PSNR={psnr(pix[0], gt[0], m[0]):.2f}  "
          f"whole-PSNR={psnr(pix[0], gt[0], torch.ones_like(m[0])):.2f}  "
          f"| vae-ceiling obj-PSNR={psnr(vae_pix[0], gt[0], m[0]):.2f}  "
          f"whole-PSNR={psnr(vae_pix[0], gt[0], torch.ones_like(m[0])):.2f}  "
          f"vid={int(b['video_id'][0])}")

    gt_u8, rec_u8, vae_u8 = to_u8(gt[0]), to_u8(pix[0]), to_u8(vae_pix[0])
    m_np = m[0].cpu().numpy()[:, 0]                                   # (T,H,W)
    rec_obj = rec_u8 * m_np[..., None].astype(np.uint8)              # objects-only recon
    S = gt_u8.shape[1]; gap = np.full((gt_u8.shape[0], S, 4, 3), 255, np.uint8)
    panels = np.concatenate([gt_u8, gap, vae_u8, gap, rec_u8, gap, rec_obj], axis=2)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), list(panels), fps=args.fps)
    print(f"[saved] {out}  (layout: GT | Wan-VAE ceiling | our reconstruction | recon x object-mask)")


if __name__ == "__main__":
    main()
