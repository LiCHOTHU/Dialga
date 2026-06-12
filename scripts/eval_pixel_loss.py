"""eval_pixel_loss.py — compute the trainer's L_pixel for a checkpoint.

Replicates EXACTLY the obs-path pixel loss logged as "pixel" in train_v5.py:

    recon_obs = dec(enc(chunk_obs))            # our model's latent recon
    pred_pix  = wan_vae.decode(recon_obs)      # NO latent normalization
    L_pixel   = MSE(pred_pix, raw_pix)         # raw_pix in [-1, 1]

This lets us compute a genuine pixel loss for checkpoints that were trained
WITHOUT pixel supervision (e.g. the old latent-recon e1), so they can be
compared apples-to-apples against e2/e3's reported `pixel` column.

Usage:
  python scripts/eval_pixel_loss.py \\
    --ckpt .../v5_best.pt --cache_dir ... --video_dir ... \\
    --n_chunks 200 --batch_size 4 --split val
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.clevrer_window_pixels import ClevrerChunkPairsWithPixels
from src.data.clevrer_window import chunk_collate
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import LatentDecoder


def load_wan_vae(model_id, dtype, device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


@torch.no_grad()
def vae_decode(vae, latent, dtype):
    """latent (B,C,T,H,W) -> pix (B,T_pix,3,H,W) in [-1,1]. Matches _vae_decode."""
    z = latent.to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out      # (B,3,T,H,W)
    return pix.permute(0, 2, 1, 3, 4).contiguous().float()


@torch.no_grad()
def vae_encode(vae, pix, dtype):
    """pix (B,T_pix,3,H,W) in [-1,1] -> latent (B,C,T_lat,H,W). Matches _vae_encode.

    Used for the encoder-unfrozen run (e2): the trainer overrides the cached
    latent by re-encoding raw pixels through the (trained) Wan encoder, so the
    eval must do the same to reflect what e2 actually learned.
    """
    x = pix.permute(0, 2, 1, 3, 4).contiguous().to(dtype)
    out = vae.encode(x)
    z = out.latent_dist.mean if hasattr(out, "latent_dist") else out.latents
    return z.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_chunks", type=int, default=200)
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_videos", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    if args.out_json is None:
        args.out_json = str(Path(args.ckpt).parent / "pixel_loss.json")

    # ---- model ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 64)); d_dyn = int(a.get("d_dyn", 64))
    enc_hidden_ch = int(a.get("enc_hidden_ch", 128))
    dec_hidden_ch = int(a.get("dec_hidden_ch", 256))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))
    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch,
                          use_layer_norm=use_ln).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_hidden_ch,
                        chunk_size_lat=chunk_size_lat).to(device)
    enc.load_state_dict(ckpt["encoder"]); dec.load_state_dict(ckpt["decoder"])
    enc.eval(); dec.eval()
    print(f"[model] {args.ckpt}")
    print(f"[model] d_static={d_static} d_dyn={d_dyn} enc_h={enc_hidden_ch} "
          f"dec_h={dec_hidden_ch} use_ln={use_ln}")

    print(f"[vae] loading {args.model_id} ({args.dtype})")
    vae = load_wan_vae(args.model_id, dtype, device)

    # If the checkpoint carries trained VAE weights (e2 enc-unfrozen / e3
    # dec-unfrozen), load them over the pretrained VAE so we measure what the
    # run actually learned. e1 (frozen VAE) has no such key -> pretrained used.
    unfreeze_enc = bool(a.get("unfreeze_vae_enc", False))
    unfreeze_dec = bool(a.get("unfreeze_vae_dec", False))
    vae_state = ckpt.get("wan_vae", None)
    if vae_state is not None:
        sd = {k: v.to(dtype) for k, v in vae_state.items()}
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        print(f"[vae] loaded TRAINED weights from ckpt "
              f"(enc_unfrozen={unfreeze_enc} dec_unfrozen={unfreeze_dec}; "
              f"missing={len(missing)} unexpected={len(unexpected)})")
    else:
        print(f"[vae] no wan_vae in ckpt -> using PRETRAINED weights "
              f"(enc_unfrozen={unfreeze_enc} dec_unfrozen={unfreeze_dec})")

    # ---- data (same dataset + split + seed as the trainer) ----
    ds = ClevrerChunkPairsWithPixels(
        args.cache_dir, args.video_dir, image_size=128,
        split=args.split, val_frac=args.val_frac,
        seed=args.seed, max_videos=args.max_videos)
    n_total = len(ds)
    n_use = min(args.n_chunks, n_total)
    g = torch.Generator().manual_seed(args.seed)
    idxs = sorted(torch.randperm(n_total, generator=g).tolist()[:n_use])
    loader = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=chunk_collate)
    print(f"[data] {args.split}: eval on {n_use}/{n_total} chunks")

    # ---- L_pixel = MSE(wan_dec(dec(enc(chunk_obs))), raw_pix) ----
    se_sum, n_elem, n_done = 0.0, 0, 0
    per_chunk = []
    t0 = time.time()
    for batch in loader:
        raw_pix = batch["pix_obs"].to(device)            # (B,T_pix,3,H,W) in [-1,1]
        # e2 path: re-encode raw pixels through the trained Wan encoder, exactly
        # as the trainer does, instead of using the cached (pretrained) latent.
        if unfreeze_enc:
            chunk_obs = vae_encode(vae, raw_pix, dtype)
        else:
            chunk_obs = batch["chunk_obs"].to(device)
        with torch.no_grad():
            out = enc(chunk_obs)
            recon = dec(out["z_static"], out["z_dyn"])
            pred_pix = vae_decode(vae, recon, dtype)
            T = min(raw_pix.shape[1], pred_pix.shape[1])
            diff = (pred_pix[:, :T] - raw_pix[:, :T]).pow(2)
            per_chunk.append(diff.mean(dim=(1, 2, 3, 4)).cpu())  # (B,)
            se_sum += diff.sum().item()
            n_elem += diff.numel()
        n_done += chunk_obs.shape[0]
        if n_done % (args.batch_size * 8) == 0 or n_done >= n_use:
            print(f"  [{n_done:4d}/{n_use}] {time.time()-t0:5.1f}s")

    per_chunk = torch.cat(per_chunk)
    l_pixel = se_sum / n_elem                       # element-mean MSE == trainer's
    print("\n" + "=" * 56)
    print(f"L_pixel (obs)  = {l_pixel:.6f}   <-- matches trainer 'pixel' column")
    print(f"per-chunk mean = {per_chunk.mean().item():.6f} "
          f"± {per_chunk.std().item():.6f}  (n={per_chunk.numel()})")
    print("=" * 56)

    out = {"ckpt": str(args.ckpt), "split": args.split, "n_chunks": n_use,
           "L_pixel_obs": l_pixel,
           "per_chunk_mean": float(per_chunk.mean()),
           "per_chunk_std": float(per_chunk.std())}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
