"""v5.8 flow-decoder isolation smoke.

Question: does a rectified-flow (generative) decoder beat the deterministic
SpatialGridDecoder's overfit reconstruction ceiling (obj-PSNR ~26-30 dB on
5-vid train)? A deterministic MSE decoder renders the conditional MEAN of the
Wan latent -> blur; a flow decoder SAMPLES a sharp latent (VideoFlexTok).

Setup: spatial-grid encoder + FlowMatchingDecoder, jointly overfit 5 vids with
the MinRF velocity loss on the OBSERVED Wan-latent chunk. No VAE decode in the
training loop. Optionally mask-weight the velocity loss by the object mask
POOLED TO THE 8x8 LATENT GRID (valid because the Wan encoder preserves spatial
position). Eval by sampling the latent, decoding through the frozen Wan VAE, and
measuring object-region / whole-frame PSNR vs GT, plus a decode-free latent
object-region recon.
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
from src.model.latent_decoder import FlowMatchingDecoder, FlowMatcher


def psnr(pred, gt, w):
    err = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean(dim=1, keepdim=True)
    mse = (err * w).sum() / w.sum().clamp(min=1.0)
    return 10.0 * math.log10(1.0 / max(mse.item(), 1e-12))


def latent_obj_mask(mask_obs, g=8):
    """(B,T,1,H,W) pixel mask -> (B,1,g,g) per-cell object occupancy in [0,1].
    Max-pool over space (128->g) and over the whole chunk in time."""
    B, T = mask_obs.shape[:2]
    m = mask_obs.reshape(B * T, 1, mask_obs.shape[-2], mask_obs.shape[-1]).float()
    m = F.adaptive_max_pool2d(m, (g, g)).reshape(B, T, 1, g, g)
    return m.amax(dim=1)                                             # (B,1,g,g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33")
    ap.add_argument("--video_dir", default="/storage/project/r-agarg35-0/lwang831/"
                    "dataset/CLEVRER/train_video")
    ap.add_argument("--out", default="/storage/scratch1/8/lwang831/smoke_v58/flow_overfit")
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_static", type=int, default=96)
    ap.add_argument("--d_dyn", type=int, default=96)
    ap.add_argument("--static_grid", type=int, default=4)
    ap.add_argument("--enc_hidden_ch", type=int, default=192)
    ap.add_argument("--dec_hidden_ch", type=int, default=384)
    ap.add_argument("--dec_depth", type=int, default=3)
    ap.add_argument("--mask_weight", type=float, default=0.0,
                    help="if >0, add this * object-region weight to the latent flow loss")
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    ds = ClevrerChunkPairsWithPixels(cache_dir=args.cache_dir, video_dir=args.video_dir,
                                     image_size=128, split="train", val_frac=0.0,
                                     seed=42, max_videos=args.max_videos, return_masks=True)
    dl = DataLoader(ds, batch_size=args.max_videos, shuffle=True, collate_fn=chunk_collate)

    enc = LatentEncoder3D(d_static=args.d_static, d_dyn=args.d_dyn,
                          hidden_ch=args.enc_hidden_ch, use_layer_norm=True,
                          pool_type="spatial", static_grid=args.static_grid).to(dev)
    dec = FlowMatchingDecoder(latent_ch=48, d_static=args.d_static, static_grid=args.static_grid,
                              d_dyn=args.d_dyn, hidden_ch=args.dec_hidden_ch,
                              chunk_size_lat=9, spatial_size=8, depth=args.dec_depth).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=1e-3)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"[flow-decoder] params={n_params/1e6:.2f}M  mask_weight={args.mask_weight}  "
          f"sample_steps={args.sample_steps}")

    vae = None  # lazy-load only at eval

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
        with torch.no_grad():
            b = next(iter(DataLoader(ds, batch_size=args.max_videos, shuffle=False,
                                     collate_fn=chunk_collate)))
            x0 = b["chunk_obs"].to(dev)
            z = enc(x0)
            x0_hat = FlowMatcher.sample(dec, z["z_static_grid"], z["z_dyn"],
                                        shape=x0.shape, steps=args.sample_steps, device=dev)
            # decode-free latent object-region recon
            lm = latent_obj_mask(b["mask_obs"].to(dev)).unsqueeze(2)     # (B,1,1,g,g)
            lat_err = (x0_hat - x0).pow(2)                                # (B,C,T,g,g)
            lat_obj = (lat_err * lm).sum() / lm.expand_as(lat_err).sum().clamp(min=1)
            lat_all = lat_err.mean()
            # pixel PSNR via Wan decode
            pix = vae.decode(x0_hat.to(torch.bfloat16))
            pix = (pix.sample if hasattr(pix, "sample") else pix).permute(0, 2, 1, 3, 4).float()
            gt = b["pix_obs"].to(dev); m = b["mask_obs"].to(dev)
            op = sum(psnr(pix[i], gt[i], m[i]) for i in range(pix.shape[0])) / pix.shape[0]
            wp = sum(psnr(pix[i], gt[i], torch.ones_like(m[i])) for i in range(pix.shape[0])) / pix.shape[0]
        print(f"[eval {tag}] sampled obj-PSNR={op:.2f}  whole-PSNR={wp:.2f}  "
              f"| latent obj-MSE={lat_obj.item():.5f}  all-MSE={lat_all.item():.5f}")
        enc.train(); dec.train()

    enc.train(); dec.train()
    for ep in range(1, args.epochs + 1):
        for b in dl:
            x0 = b["chunk_obs"].to(dev)
            z = enc(x0)
            x_sig, sigma, v_tgt = FlowMatcher.add_noise(x0)
            v_pred = dec(x_sig, sigma, z["z_static_grid"], z["z_dyn"])
            se = (v_pred - v_tgt).pow(2)
            if args.mask_weight > 0:
                lm = latent_obj_mask(b["mask_obs"].to(dev)).unsqueeze(2)  # (B,1,1,g,g)
                w = 1.0 + args.mask_weight * lm
                loss = (se * w).sum() / w.expand_as(se).sum()
            else:
                loss = se.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == 1:
            print(f"ep {ep:4d}  flow_loss={loss.item():.5f}")
        if ep % args.eval_every == 0:
            evaluate(f"ep{ep}")

    evaluate("final")
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                "args": vars(args), "epoch": args.epochs},
               str(Path(args.out) / "last.pt"))
    print("ALL_DONE_FLOW_OVERFIT")


if __name__ == "__main__":
    main()
