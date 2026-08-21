"""Decisive gate for the v6 flow-decoder premise: deterministic vs generative
decoder on VALIDATION obj-PSNR (not overfit).

A deterministic MSE decoder renders the conditional MEAN of the val latent ->
blur; a rectified-flow decoder samples a sharp latent. Overfit hides this (no
ambiguity to average). So we train BOTH decoders on the SAME spatial-grid
encoder + SAME train data, and compare VAL object-region PSNR. Encoder+decoder
are trained jointly per arm (end-to-end, how they'd be used); the only variable
is decoder type. Object mask pooled to the 8x8 latent grid gives a decode-free
object-region loss/metric (Wan encoder preserves spatial position).

Run twice: --decoder det  and  --decoder flow. Same seed/data => apples-to-apples.
"""
import argparse, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.data.clevrer_window import chunk_collate
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import SpatialGridDecoder, FlowMatchingDecoder, FlowMatcher


def psnr(pred, gt, w):
    err = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean(dim=1, keepdim=True)
    mse = (err * w).sum() / w.sum().clamp(min=1.0)
    return 10.0 * math.log10(1.0 / max(mse.item(), 1e-12))


def latent_obj_mask(mask_obs, g=8):
    B, T = mask_obs.shape[:2]
    m = mask_obs.reshape(B * T, 1, mask_obs.shape[-2], mask_obs.shape[-1]).float()
    m = F.adaptive_max_pool2d(m, (g, g)).reshape(B, T, 1, g, g)
    return m.amax(dim=1)                                              # (B,1,g,g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=["det", "flow"], required=True)
    ap.add_argument("--cache_dir", default="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33")
    ap.add_argument("--video_dir", default="/storage/project/r-agarg35-0/lwang831/"
                    "dataset/CLEVRER/train_video")
    ap.add_argument("--out", default="/storage/scratch1/8/lwang831/v6_headtohead")
    ap.add_argument("--max_videos", type=int, default=500)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_static", type=int, default=96)
    ap.add_argument("--d_dyn", type=int, default=96)
    ap.add_argument("--static_grid", type=int, default=4)
    ap.add_argument("--enc_hidden_ch", type=int, default=192)
    ap.add_argument("--dec_hidden_ch", type=int, default=384)
    ap.add_argument("--dec_depth", type=int, default=3)
    ap.add_argument("--mask_weight", type=float, default=2.0,
                    help="object-region upweight on the (latent) recon loss")
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--n_val_clips", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.out) / args.decoder
    outdir.mkdir(parents=True, exist_ok=True)

    def make_ds(split):
        return ClevrerChunkPairsWithPixels(cache_dir=args.cache_dir, video_dir=args.video_dir,
                                           image_size=128, split=split, val_frac=args.val_frac,
                                           seed=42, max_videos=args.max_videos, return_masks=True)
    tr = make_ds("train"); va = make_ds("val")
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    collate_fn=chunk_collate, num_workers=4, drop_last=True)
    vdl = DataLoader(va, batch_size=args.batch_size, shuffle=False, collate_fn=chunk_collate)
    print(f"[data] decoder={args.decoder}  train_chunks={len(tr)}  val_chunks={len(va)}")

    enc = LatentEncoder3D(d_static=args.d_static, d_dyn=args.d_dyn,
                          hidden_ch=args.enc_hidden_ch, use_layer_norm=True,
                          pool_type="spatial", static_grid=args.static_grid).to(dev)
    if args.decoder == "det":
        dec = SpatialGridDecoder(latent_ch=48, d_static=args.d_static, static_grid=args.static_grid,
                                 d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                                 chunk_size_lat=9, spatial_size=8, depth=args.dec_depth).to(dev)
    else:
        dec = FlowMatchingDecoder(latent_ch=48, d_static=args.d_static, static_grid=args.static_grid,
                                  d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                                  chunk_size_lat=9, spatial_size=8, depth=args.dec_depth).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=1e-3)
    print(f"[model] dec_params={sum(p.numel() for p in dec.parameters())/1e6:.2f}M")

    vae = None

    def evaluate(tag):
        nonlocal vae
        if vae is None:
            from diffusers import AutoencoderKLWan
            vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                                   subfolder="vae",
                                                   torch_dtype=torch.bfloat16).eval().to(dev)
            for p in vae.parameters():
                p.requires_grad_(False)
        enc.eval(); dec.eval()
        ops, wps, seen = 0.0, 0.0, 0
        lat_obj_sum, lat_n = 0.0, 0
        with torch.no_grad():
            for b in vdl:
                x0 = b["chunk_obs"].to(dev); z = enc(x0)
                if args.decoder == "det":
                    x0_hat = dec(z["z_static_grid"], z["z_dyn"])
                else:
                    x0_hat = FlowMatcher.sample(dec, z["z_static_grid"], z["z_dyn"],
                                                shape=x0.shape, steps=args.sample_steps, device=dev)
                lm = latent_obj_mask(b["mask_obs"].to(dev)).unsqueeze(2)
                le = (x0_hat - x0).pow(2)
                lat_obj_sum += (le * lm).sum().item() / max(lm.expand_as(le).sum().item(), 1)
                lat_n += 1
                pix = vae.decode(x0_hat.to(torch.bfloat16))
                pix = (pix.sample if hasattr(pix, "sample") else pix).permute(0, 2, 1, 3, 4).float()
                gt = b["pix_obs"].to(dev); m = b["mask_obs"].to(dev)
                for i in range(pix.shape[0]):
                    if seen >= args.n_val_clips:
                        break
                    ops += psnr(pix[i], gt[i], m[i])
                    wps += psnr(pix[i], gt[i], torch.ones_like(m[i]))
                    seen += 1
                if seen >= args.n_val_clips:
                    break
        print(f"[val {tag}] obj-PSNR={ops/seen:.2f}  whole-PSNR={wps/seen:.2f}  "
              f"latent obj-MSE={lat_obj_sum/lat_n:.5f}  (n={seen})", flush=True)
        enc.train(); dec.train()

    enc.train(); dec.train()
    for ep in range(1, args.epochs + 1):
        for b in dl:
            x0 = b["chunk_obs"].to(dev); z = enc(x0)
            lm = latent_obj_mask(b["mask_obs"].to(dev)).unsqueeze(2)
            w = 1.0 + args.mask_weight * lm
            if args.decoder == "det":
                x0_hat = dec(z["z_static_grid"], z["z_dyn"])
                se = (x0_hat - x0).pow(2)
            else:
                x_sig, sigma, v_tgt = FlowMatcher.add_noise(x0)
                v_pred = dec(x_sig, sigma, z["z_static_grid"], z["z_dyn"])
                se = (v_pred - v_tgt).pow(2)
            loss = (se * w).sum() / w.expand_as(se).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:4d}  loss={loss.item():.5f}", flush=True)
        if ep % args.eval_every == 0:
            evaluate(f"ep{ep}")

    evaluate("final")
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                "args": vars(args), "epoch": args.epochs}, str(outdir / "last.pt"))
    print(f"ALL_DONE_{args.decoder.upper()}", flush=True)


if __name__ == "__main__":
    main()
