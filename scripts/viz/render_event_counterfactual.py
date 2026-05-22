"""scripts/viz/render_event_counterfactual.py — TODO 4.

For ~10 val collision chunks, render two side-by-side pixel videos:
    (a) no-event rollout :  z_dyn_pred = fwd.chunk_step(z_dyn_last)
    (b) with-event       :  z_dyn_pred = (a) + g_event(eh(z_dyn_last))

The "GT" reference is the Wan-VAE roundtrip of the cached chunk_pred latent
(same ceiling that any rollout-then-decode is bounded by).

Outputs:
    <out_dir>/cf_<idx>_<vid>_<start>.mp4   GT | no-event | with-event
    <out_dir>/event_counterfactual.json    residual norms + per-chunk PSNR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.event_head import EventHead, GEvent
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


@torch.no_grad()
def wan_decode(vae, wan_latent, dtype):
    """wan_latent: (B, 48, T_lat, 8, 8) -> pixels (B, T_pix, 3, H, W) in [-1, 1]."""
    z = wan_latent.to(dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    return pix.float().permute(0, 2, 1, 3, 4).contiguous()


def to_uint8(x):
    """[-1, 1] -> uint8 (H, W, 3)."""
    return ((x.clamp(-1, 1) + 1) * 0.5 * 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def stack_h(*frames):
    """List of (H,W,3) np.uint8 -> single (H, W*N + sep, 3) with white separators."""
    h = frames[0].shape[0]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    out = []
    for i, f in enumerate(frames):
        if i: out.append(sep)
        out.append(f)
    return np.concatenate(out, axis=1)


def psnr_pix(a, b, max_val: float = 2.0) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10.0 * float(np.log10((max_val ** 2) / max(mse, 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="v5_events.pt")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_chunks", type=int, default=10)
    ap.add_argument("--out_dir", default=None,
                    help="default: <ckpt parent>/counterfactual_render")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[args.dtype]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn    = int(a.get("d_dyn", 16))
    d_state  = int(a.get("d_state", 8))
    d_event  = int(a.get("d_event", 4))
    enc_hid  = int(a.get("enc_hidden_ch", 32))
    dec_hid  = int(a.get("dec_hidden_ch", 64))
    chunk_T  = int(a.get("chunk_size_lat", 9))
    no_proj  = bool(a.get("no_proj", False))
    shared_trunk = bool(a.get("shared_trunk", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn,
                          hidden_ch=enc_hid, shared_trunk=shared_trunk).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn,
                        hidden_ch=dec_hid, chunk_size_lat=chunk_T).to(device)
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    eh = EventHead(d_dyn=d_dyn, d_event=d_event).to(device)
    ge = GEvent(d_event=d_event, d_dyn=d_dyn).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    fwd.load_state_dict(ckpt["fwd"])
    eh.load_state_dict(ckpt["event_head"])
    ge.load_state_dict(ckpt["g_event"])
    for m in (enc, dec, fwd, eh, ge): m.eval()
    print(f"[model] events ckpt loaded from {args.ckpt}")

    print(f"[vae] loading {args.model_id}")
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    vae = vae.to(device)

    ds = ClevrerChunkPairs(args.cache_dir, split="val",
                            val_frac=args.val_frac, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=chunk_collate)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "counterfactual_render"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rendered = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            gate_GT = batch["gate_GT"]
            for i in range(gate_GT.shape[0]):
                if rendered >= args.n_chunks:
                    break
                if gate_GT[i].item() < 0.5:
                    continue   # skip non-collision chunks
                chunk_obs  = batch["chunk_obs"][i:i+1].to(device)
                chunk_pred = batch["chunk_pred"][i:i+1].to(device)
                vid = int(batch["video_id"][i].item())
                s   = int(batch["start_frame"][i].item())

                enc_obs = enc(chunk_obs)
                z_static = enc_obs["z_static"]
                z_dyn_obs = enc_obs["z_dyn"]
                z_dyn_last = z_dyn_obs[:, -1]
                T = z_dyn_obs.shape[1]
                z_dyn_base = fwd.chunk_step(z_dyn_last, T)
                correction = ge(eh(z_dyn_last)).unsqueeze(1).expand(-1, T, -1)
                z_dyn_no_event   = z_dyn_base
                z_dyn_with_event = z_dyn_base + correction
                residual_norm = float((z_dyn_with_event - z_dyn_no_event).norm(dim=-1).mean())

                wan_no   = dec(z_static, z_dyn_no_event)
                wan_with = dec(z_static, z_dyn_with_event)

                pix_GT   = wan_decode(vae, chunk_pred, dtype)[0].cpu()   # (T_pix, 3, H, W)
                pix_no   = wan_decode(vae, wan_no, dtype)[0].cpu()
                pix_with = wan_decode(vae, wan_with, dtype)[0].cpu()

                psnr_no   = psnr_pix(pix_no.numpy(),   pix_GT.numpy())
                psnr_with = psnr_pix(pix_with.numpy(), pix_GT.numpy())

                # Render side-by-side video
                T_pix = pix_GT.shape[0]
                frames = []
                for t in range(T_pix):
                    frames.append(stack_h(
                        to_uint8(pix_GT[t]),
                        to_uint8(pix_no[t]),
                        to_uint8(pix_with[t]),
                    ))
                fname = out_dir / f"cf_{rendered:02d}_vid{vid:05d}_start{s:03d}.mp4"
                if imageio is not None:
                    imageio.mimwrite(str(fname), frames, fps=8, codec="libx264")
                else:
                    # Fallback: save first / mid / last as PNGs
                    import PIL.Image as Image
                    for j, ti in enumerate([0, T_pix // 2, T_pix - 1]):
                        Image.fromarray(frames[ti]).save(str(fname).replace(".mp4", f"_t{j}.png"))

                results.append({
                    "idx": rendered, "video_id": vid, "start_frame": s,
                    "residual_norm": residual_norm,
                    "psnr_no_event": psnr_no,
                    "psnr_with_event": psnr_with,
                    "psnr_gain": psnr_with - psnr_no,
                    "file": str(fname),
                })
                print(f"  [{rendered:02d}] vid={vid} start={s} "
                      f"residual={residual_norm:.4f}  psnr no/with = {psnr_no:.2f} / {psnr_with:.2f}  "
                      f"(+{psnr_with-psnr_no:+.3f}) -> {fname.name}")
                rendered += 1
            if rendered >= args.n_chunks:
                break

    if results:
        norms = [r["residual_norm"] for r in results]
        gains = [r["psnr_gain"] for r in results]
        print(f"\n[summary] rendered {len(results)} collision chunks in {time.time()-t0:.1f}s")
        print(f"  mean residual_norm: {np.mean(norms):.4f}   max: {np.max(norms):.4f}")
        print(f"  mean psnr_gain (with−no): {np.mean(gains):+.3f}   max: {np.max(gains):+.3f}")
    out_json = out_dir / "event_counterfactual.json"
    out_json.write_text(json.dumps({"results": results, "args": vars(args)}, indent=2))
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
