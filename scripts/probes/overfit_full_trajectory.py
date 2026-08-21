"""Overfit the decoder on ONE full CLEVRER trajectory, then reconstruct the
whole thing — a clearer demo than a single chunk.

A video is cached as 3 consecutive 33-frame chunks (start frames 0/33/66 = 99
frames). We freeze the encoder, fine-tune ONLY the decoder to reconstruct all 3
chunks of one video with the object-centric mask-weighted pixel loss
(L = lambda_recon*MSE_latent + lambda_pixel*weighted_pixel_MSE,
 w = bg_weight + (1-bg_weight)*object_mask), then decode all 3 chunks and stitch
them into a 99-frame reconstruction.

Output: outputs/full_traj_recon_v<ID>.mp4 = [ GT | reconstruction | recon x mask ]
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from diffusers import AutoencoderKLWan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.model.latent_decoder import LatentDecoder, SpatialBroadcastDecoder, SlotDecoder
from src.model.latent_encoder import LatentEncoder3D


def build_enc(a, state, dev):
    use_ln = "norm_static.weight" in state
    enc = LatentEncoder3D(d_static=int(a["d_static"]), d_dyn=int(a["d_dyn"]),
                          hidden_ch=int(a["enc_hidden_ch"]), use_layer_norm=use_ln,
                          pool_type=a.get("pool_type", "mean"),
                          n_queries=int(a.get("pool_queries", 8)),
                          n_heads=int(a.get("pool_heads", 4))).to(dev)
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
    return LatentDecoder(**kw).to(dev)


def to_u8(x):  # (T,3,H,W) [-1,1] -> (T,H,W,3) uint8
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
    ap.add_argument("--video_id", type=int, default=-1, help="-1 = first cached video")
    ap.add_argument("--bg_weight", type=float, default=0.1)
    ap.add_argument("--lambda_recon", type=float, default=0.25)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/full_traj_recon.mp4")
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
    dec = build_dec(a, dev); dec.load_state_dict(ck["decoder"])
    is_slot = isinstance(dec, SlotDecoder)

    cache_dir = Path(a["cache_dir"])
    windows = json.loads((cache_dir / "metadata.json").read_text())["windows"]
    by_vid = {}
    for i, w in enumerate(windows):
        by_vid.setdefault(int(w["video_id"]), []).append(i)
    by_vid = {v: sorted(ws, key=lambda i: int(windows[i]["start_frame"]))
              for v, ws in by_vid.items() if len(ws) >= 3}
    vid = args.video_id if args.video_id >= 0 else sorted(by_vid)[0]
    idxs = by_vid[vid][:3]
    print(f"[ckpt] {args.ckpt} dec={a.get('decoder_type')} pool={a.get('pool_type')} "
          f"| video={vid} chunks={[int(windows[i]['start_frame']) for i in idxs]} bg_weight={args.bg_weight}")

    # helper instance only for pixel/mask loaders
    helper = ClevrerChunkPairsWithPixels(cache_dir=str(cache_dir), video_dir=args.video_dir,
                                         image_size=128, return_masks=True)
    chunks = []
    for i in idxs:
        blob = torch.load(cache_dir / windows[i]["path"], map_location="cpu", weights_only=False)
        sf = int(blob["start_frame"])
        chunks.append(dict(
            lat=blob["latent"].unsqueeze(0).to(dev),                       # (1,48,9,8,8)
            pix=helper._load_chunk_pixels(vid, sf).unsqueeze(0).to(dev),   # (1,33,3,128,128)
            msk=helper._load_chunk_masks(vid, sf).unsqueeze(0).to(dev)))   # (1,33,1,128,128)

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                           subfolder="vae", torch_dtype=vdt).eval().to(dev)
    for p in vae.parameters():
        p.requires_grad_(False)

    def vae_decode(lat):
        out = vae.decode(lat.to(vdt))
        return (out.sample if hasattr(out, "sample") else out).permute(0, 2, 1, 3, 4).float()

    def cond(z):
        return z["z_slots"] if is_slot else z["z_static"]

    @torch.no_grad()
    def eval_traj():
        dec.eval(); fg = wh = 0.0
        for c in chunks:
            z = enc(c["lat"]); pred = vae_decode(dec(cond(z), z["z_dyn"]))
            fg += psnr(pred[0], c["pix"][0], c["msk"][0])
            wh += psnr(pred[0], c["pix"][0], torch.ones_like(c["msk"][0]))
        return fg / len(chunks), wh / len(chunks)

    fg0, wh0 = eval_traj()
    print(f"[baseline]  obj-PSNR={fg0:.2f}  whole-PSNR={wh0:.2f}")

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        dec.train(); run = 0.0
        for c in chunks:
            z = enc(c["lat"])
            rec_lat = dec(cond(z), z["z_dyn"])
            L_recon = F.mse_loss(rec_lat, c["lat"])
            pred = vae_decode(rec_lat)
            w = args.bg_weight + (1.0 - args.bg_weight) * c["msk"]
            err = (pred - c["pix"]).pow(2).mean(dim=2, keepdim=True)
            L_pixel = (err * w).sum() / w.sum().clamp(min=1.0)
            loss = args.lambda_recon * L_recon + args.lambda_pixel * L_pixel
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item()
        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            fg, wh = eval_traj()
            print(f"ep{ep:>3} loss={run/len(chunks):.4f}  obj-PSNR={fg:.2f}  "
                  f"whole-PSNR={wh:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # stitch full-trajectory reconstruction
    dec.eval()
    gt_all, rec_all, recobj_all = [], [], []
    with torch.no_grad():
        for c in chunks:
            z = enc(c["lat"]); pred = vae_decode(dec(cond(z), z["z_dyn"]))
            gt_u8, rec_u8 = to_u8(c["pix"][0]), to_u8(pred[0])
            m = c["msk"][0].cpu().numpy()[:, 0]
            gt_all.append(gt_u8); rec_all.append(rec_u8)
            recobj_all.append(rec_u8 * m[..., None].astype(np.uint8))
    gt = np.concatenate(gt_all); rec = np.concatenate(rec_all); recobj = np.concatenate(recobj_all)
    S = gt.shape[1]; gap = np.full((gt.shape[0], S, 4, 3), 255, np.uint8)
    panels = np.concatenate([gt, gap, rec, gap, recobj], axis=2)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), list(panels), fps=args.fps)
    fg, wh = eval_traj()
    print(f"[done] {args.epochs} ep: obj-PSNR {fg0:.2f}->{fg:.2f} ({fg-fg0:+.2f} dB), "
          f"whole {wh0:.2f}->{wh:.2f}")
    print(f"[saved] {out}  ({gt.shape[0]} frames, layout: GT | reconstruction | recon x mask)")


if __name__ == "__main__":
    main()
