"""Save side-by-side mp4s of GT vs model reconstruction+rollout for v5.1 overfit.

For each of N videos:
  - Load two adjacent chunks (chunk_obs, chunk_pred) from the v5.1 cache.
  - Encode chunk_obs through the trained v5.1 encoder -> z_static_a, z_dyn_obs.
  - Decode the OBSERVED chunk:   dec(z_static_a, z_dyn_obs).
  - Roll forward:                z_dyn_roll = fwd.step(z_dyn_obs) + gate*g_event(eh(z_dyn_obs))
  - Decode the PREDICTED chunk:  dec(z_static_a, z_dyn_roll).
  - Pipe both GT and reconstructed latents through Wan VAE decoder -> pixels.
  - Write an mp4 with GT on the left, model on the right, time across the whole
    pair (chunk_obs frames then chunk_pred frames), at the model-internal
    128x128 resolution.

Usage:
    python scripts/save_v51_overfit_videos.py \\
        --ckpt outputs/v51_overfit_20vid_<stamp>/v5.pt \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_20vid_W33_local \\
        --n_videos 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
def wan_decode(vae, latent, device, dtype) -> torch.Tensor:
    """latent: (C, T_lat, H, W). Returns (T_pix, 3, H_pix, W_pix) in [-1, 1]."""
    z = latent.unsqueeze(0).to(device).to(dtype)        # (1, C, T_lat, H, W)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out
    # diffusers Wan returns (1, 3, T, H, W); permute to (T, 3, H, W)
    pix = pix.squeeze(0).permute(1, 0, 2, 3).contiguous()
    return pix.float().cpu()


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> uint8 [0, 255]. Input (T, 3, H, W) -> (T, H, W, 3)."""
    x = (x.clamp(-1, 1) + 1) * 0.5 * 255.0
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_videos", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="defaults to <ckpt parent>/rollouts")
    ap.add_argument("--model_id", type=str,
                    default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--use_gt_gate", action="store_true",
                    help="Use cached gate_GT (True for collisions). Default uses "
                         "the trained GatePredictor's sigmoid output > 0.5.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load v5.1 model ----
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn = int(a.get("d_dyn", 16))
    d_state = int(a.get("d_state", 8))
    d_event = int(a.get("d_event", 4))
    enc_hidden_ch = int(a.get("enc_hidden_ch", 32))
    dec_hidden_ch = int(a.get("dec_hidden_ch", 64))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))
    no_proj = bool(a.get("no_proj", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hidden_ch).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn, hidden_ch=dec_hidden_ch,
                        chunk_size_lat=chunk_size_lat).to(device)
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    eh = EventHead(d_dyn=d_dyn, d_event=d_event).to(device)
    ge = GEvent(d_event=d_event, d_dyn=d_dyn).to(device)
    gp = GatePredictor(d_dyn=d_dyn).to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    fwd.load_state_dict(ckpt["fwd"])
    eh.load_state_dict(ckpt["event_head"])
    ge.load_state_dict(ckpt["g_event"])
    gp.load_state_dict(ckpt["gate_predictor"])
    for m in (enc, dec, fwd, eh, ge, gp): m.eval()
    print(f"[model] v5.1 loaded from {args.ckpt}")

    # ---- load Wan VAE for pixel decoding ----
    print(f"[vae] loading {args.model_id}")
    vae = load_wan_vae(args.model_id, dtype, device)
    print(f"[vae] loaded; {sum(p.numel() for p in vae.parameters())/1e6:.1f}M params")

    # ---- load chunks ----
    cache_dir = Path(args.cache_dir)
    meta = json.loads((cache_dir / "metadata.json").read_text())
    windows = meta["windows"]
    by_vid: dict[int, list[int]] = {}
    for i, w in enumerate(windows):
        by_vid.setdefault(int(w["video_id"]), []).append(i)
    for v in by_vid:
        by_vid[v] = sorted(by_vid[v], key=lambda i: int(windows[i]["start_frame"]))

    vid_ids = sorted(by_vid.keys())[: args.n_videos]
    print(f"[data] writing rollouts for {len(vid_ids)} videos: {vid_ids}")

    @torch.no_grad()
    def model_recon(chunk_obs_latent: torch.Tensor, gate_GT_scalar: float):
        """chunk_obs_latent: (C, T_lat, H, W). Returns (recon_obs_latent, recon_pred_latent)."""
        z = chunk_obs_latent.unsqueeze(0).to(device)
        out = enc(z)
        z_static, z_dyn_obs = out["z_static"], out["z_dyn"]    # z_dyn: (1, T, D_d)
        T_chunk = z_dyn_obs.shape[1]
        z_dyn_last = z_dyn_obs[:, -1]                          # (1, D_d)
        z_dyn_base = fwd.chunk_step(z_dyn_last, T_chunk)       # (1, T, D_d) — one chunk-to-chunk call
        if args.use_gt_gate:
            gate = torch.tensor([gate_GT_scalar], device=device, dtype=z_dyn_obs.dtype)
        else:
            gate = (torch.sigmoid(gp(z_dyn_last)) > 0.5).float()
        correction = ge(eh(z_dyn_last))                        # (1, D_d)
        correction_t = correction.unsqueeze(1).expand(-1, T_chunk, -1)
        z_dyn_roll = z_dyn_base + correction_t * gate.unsqueeze(-1).unsqueeze(-1)
        recon_obs = dec(z_static, z_dyn_obs).squeeze(0)
        recon_pred = dec(z_static, z_dyn_roll).squeeze(0)
        return recon_obs.cpu(), recon_pred.cpu()

    for vid in vid_ids:
        ws = by_vid[vid]
        # pair A: obs=ws[0], pred=ws[1]
        obs_blob = torch.load(cache_dir / windows[ws[0]]["path"],
                              map_location="cpu", weights_only=False)
        pred_blob = torch.load(cache_dir / windows[ws[1]]["path"],
                               map_location="cpu", weights_only=False)
        chunk_obs_latent = obs_blob["latent"]                # (C, T_lat, H, W)
        chunk_pred_latent = pred_blob["latent"]
        gate_GT = float(pred_blob["collision_mask"].any().item())

        recon_obs, recon_pred = model_recon(chunk_obs_latent, gate_GT)

        # decode all four through Wan
        gt_obs_pix    = wan_decode(vae, chunk_obs_latent, device, dtype)
        gt_pred_pix   = wan_decode(vae, chunk_pred_latent, device, dtype)
        rec_obs_pix   = wan_decode(vae, recon_obs,         device, dtype)
        rec_pred_pix  = wan_decode(vae, recon_pred,        device, dtype)

        # stack: top row GT (obs then pred), bottom row Model (obs then pred)
        gt_pix    = torch.cat([gt_obs_pix,    gt_pred_pix],   dim=0)   # (T_pix*2, 3, H, W)
        model_pix = torch.cat([rec_obs_pix,   rec_pred_pix],  dim=0)
        # side-by-side per frame: (T_pix*2, 3, H, 2W)
        T_total = gt_pix.shape[0]
        sbs = torch.cat([gt_pix, model_pix], dim=-1)                   # width-wise
        sbs_u8 = to_uint8(sbs)                                         # (T, H, 2W, 3)

        out_path = out_dir / f"video_{vid:05d}_gt_vs_model.mp4"
        imageio.mimwrite(out_path, sbs_u8.numpy(), fps=8, codec="libx264", quality=8)
        gate_pred = float(torch.sigmoid(
            gp(enc(chunk_obs_latent.unsqueeze(0).to(device))["z_dyn"][:, -1])
        ).item())
        boundary_frame = gt_obs_pix.shape[0]
        print(f"  vid={vid:5d}  gate_GT={gate_GT:.0f}  gate_pred_p={gate_pred:.3f}  "
              f"frames={T_total} (boundary at {boundary_frame})  -> {out_path.name}")

    print(f"\n[done] wrote {len(vid_ids)} videos to {out_dir}")
    print("Layout: GT (left) | Model (right). First half = chunk_obs, second half = chunk_pred.")


if __name__ == "__main__":
    main()
