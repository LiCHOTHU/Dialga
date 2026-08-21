"""Object-centric pixel reconstruction: fine-tune ONLY the decoder of a frozen
model with an object-mask-WEIGHTED pixel loss, to test the user's idea that
up-weighting object pixels (and down-weighting the dominant smooth background)
sharpens objects despite the global-pool encoder.

Setup: freeze encoder (mean@ep190) + its trained decoder is the START point.
Decode decoder output through the FROZEN Wan VAE to pixels; loss =
  lambda_recon * MSE_latent(dec(z), wan_latent)                  (stabiliser)
  + lambda_pixel * weighted_pixel_MSE(vae_dec(dec(z)), pixels, mask)
where weighted = sum[w*(pred-gt)^2]/sum[w], w = bg_weight + (1-bg_weight)*mask.

bg_weight=1.0 -> uniform pixel loss (CONTROL); bg_weight<1 -> object-centric.

Metric that matters: OBJECT-REGION PSNR (PSNR over foreground pixels only) on
val. Whole-frame PSNR barely moves because background is ~94% of pixels.
Reports original-decoder vs fine-tuned, foreground AND whole-frame.
"""
import argparse, math, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKLWan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.data.clevrer_window import chunk_collate
from src.model.latent_decoder import LatentDecoder, SpatialBroadcastDecoder, SlotDecoder
from src.model.latent_encoder import LatentEncoder3D


def build_enc(a, state, device):
    use_ln = "norm_static.weight" in state
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=a.get("pool_type", "mean"),
                          n_queries=int(a.get("pool_queries", 8)),
                          n_heads=int(a.get("pool_heads", 4))).to(device)
    enc.load_state_dict(state); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_dec(a, device):
    dt = a.get("decoder_type", "linear")
    kw = dict(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
              hidden_ch=int(a["dec_hidden_ch"]),
              chunk_size_lat=int(a.get("chunk_size_lat", 9)))
    if dt == "broadcast":
        return SpatialBroadcastDecoder(depth=int(a.get("dec_depth", 3)), **kw).to(device)
    if dt == "slot":
        return SlotDecoder(depth=int(a.get("dec_depth", 3)), **kw).to(device)
    return LatentDecoder(**kw).to(device)


def masked_psnr(pred, gt, w):
    """PSNR over pixels weighted by w (w=mask -> foreground PSNR)."""
    err = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean(dim=1, keepdim=True)
    mse = (err * w).sum() / w.sum().clamp(min=1.0)
    return 10.0 * math.log10(1.0 / max(mse.item(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg_weight", type=float, default=0.1,
                    help="background pixel weight (1.0 = uniform control)")
    ap.add_argument("--lambda_recon", type=float, default=0.25)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_videos", type=int, default=400)
    ap.add_argument("--val_batches", type=int, default=30)
    ap.add_argument("--eval_on_train", action="store_true",
                    help="evaluate obj-PSNR on the TRAIN clips (overfit smoke test)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vae_dtype", type=str, default="bfloat16")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32}[args.vae_dtype]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
    enc = build_enc(a, ck["encoder"], dev)
    dec = build_dec(a, dev)
    dec.load_state_dict(ck["decoder"])          # START from the trained decoder
    is_slot = isinstance(dec, SlotDecoder)
    print(f"[ckpt] {args.ckpt} ep={ck.get('epoch')} pool={a.get('pool_type','mean')} "
          f"dec={a.get('decoder_type','linear')} | bg_weight={args.bg_weight}")

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                           subfolder="vae", torch_dtype=vdt).eval().to(dev)
    for p in vae.parameters():
        p.requires_grad_(False)
    try:                                  # recompute VAE-decode activations in backward
        vae.enable_gradient_checkpointing()   # (grad still flows to the latent input)
        print("[vae] gradient checkpointing ON")
    except Exception as e:
        print(f"[vae] no gradient checkpointing ({e})")

    def vae_decode(latent):  # (B,48,9,8,8) -> (B,33,3,128,128), grad flows to latent
        out = vae.decode(latent.to(vdt))
        pix = out.sample if hasattr(out, "sample") else out      # (B,3,T,H,W)
        return pix.permute(0, 2, 1, 3, 4).float()                # (B,T,3,H,W)

    common = dict(cache_dir=a["cache_dir"], video_dir=args.video_dir, image_size=128,
                  val_frac=float(a.get("val_frac", 0.2)), seed=int(a.get("seed", 42)),
                  max_videos=args.max_videos, return_masks=True)
    tr = ClevrerChunkPairsWithPixels(split="train", **common)
    va = ClevrerChunkPairsWithPixels(split="train" if args.eval_on_train else "val",
                                     **common)
    tdl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                     collate_fn=chunk_collate, drop_last=True)
    vdl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4,
                     collate_fn=chunk_collate)
    print(f"[data] train={len(tr)} val={len(va)} pairs")

    def cond(z):
        return z["z_slots"] if is_slot else z["z_static"]

    @torch.no_grad()
    def evaluate():
        dec.eval()
        fg = whole = n = 0.0
        for bi, b in enumerate(vdl):
            if bi >= args.val_batches:
                break
            x = b["chunk_obs"].to(dev); z = enc(x)
            pred = vae_decode(dec(cond(z), z["z_dyn"]))           # (B,T,3,H,W)
            gt = b["pix_obs"].to(dev); m = b["mask_obs"].to(dev)  # (B,T,3,..),(B,T,1,..)
            B = x.size(0)
            fg += masked_psnr(pred, gt, m) * B
            whole += masked_psnr(pred, gt, torch.ones_like(m)) * B
            n += B
        dec.train()
        return fg / n, whole / n

    fg0, wh0 = evaluate()
    print(f"[baseline orig decoder]  obj-PSNR={fg0:.2f}  whole-PSNR={wh0:.2f}")

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr)
    best_fg = fg0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); run = nb = 0.0
        for b in tdl:
            x = b["chunk_obs"].to(dev)
            with torch.no_grad():
                z = enc(x)
            rec_lat = dec(cond(z), z["z_dyn"])
            L_recon = F.mse_loss(rec_lat, x)
            pred = vae_decode(rec_lat)
            gt = b["pix_obs"].to(dev); m = b["mask_obs"].to(dev)
            w = args.bg_weight + (1.0 - args.bg_weight) * m
            err = (pred - gt).pow(2).mean(dim=2, keepdim=True)    # over RGB
            L_pixel = (err * w).sum() / w.sum().clamp(min=1.0)
            loss = args.lambda_recon * L_recon + args.lambda_pixel * L_pixel
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        fg, wh = evaluate()
        tag = "  *" if fg > best_fg else ""
        best_fg = max(best_fg, fg)
        print(f"ep{ep:>3} loss={run/nb:.4f}  obj-PSNR={fg:.2f}  whole-PSNR={wh:.2f}"
              f"  ({time.time()-t0:.0f}s){tag}", flush=True)
        if fg >= best_fg:
            sa = dict(a); sa.update(decoder_type=a.get("decoder_type", "linear"))
            torch.save({"encoder": ck["encoder"], "decoder": dec.state_dict(),
                        "args": sa, "epoch": ep, "bg_weight": args.bg_weight,
                        "obj_psnr": fg, "whole_psnr": wh}, args.out)

    print(f"\n[done] orig obj-PSNR={fg0:.2f} -> best obj-PSNR={best_fg:.2f} "
          f"(delta {best_fg-fg0:+.2f} dB), bg_weight={args.bg_weight}")


if __name__ == "__main__":
    main()
