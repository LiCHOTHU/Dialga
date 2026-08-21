"""DROID camera-conditioning readout — the v5.8 moving-camera paper figure.

On real DROID wrist-camera video there is NO un-warped reference (we never see
the scene from a canonical viewpoint), so the synthetic camera-invariance loss
(paired warped/un-warped chunks) is inapplicable. Instead we read out camera
conditioning OFFLINE from the learned codes of a trained checkpoint:

  R1. z_dyn vs camera velocity.
      The wrist camera is EE-mounted; almost all apparent motion in the video is
      the camera moving over a (largely static) tabletop. If pose conditioning
      works, the encoder DIVIDES OUT the known camera trajectory, so z_dyn should
      carry little of it:
        - corr(‖z_dyn‖_frame, ‖cam_vel‖_frame)  -> LOWER (nearer 0) is better.
        - R² of a linear map  z_dyn(frame) -> cam_vel(frame)  -> LOWER is better
          (less camera motion is linearly recoverable from z_dyn).
      The pose-BLIND control never sees pose, so camera motion has nowhere to go
      but into z_dyn -> both numbers should be HIGHER for BLIND.

  R2. z_static viewpoint stability.
      Within one episode the scene identity is fixed while the viewpoint sweeps.
      A viewpoint-invariant static code is stable across an episode's chunks:
        - mean within-episode std of z_static (averaged over dims) -> LOWER better.
      Reported as a ratio to the across-episode std (so 0 = perfectly stable
      within an episode, 1 = within-episode spread as large as between scenes).

Each checkpoint is evaluated with the pose input it was TRAINED with (the BLIND
control is fed zeros, matching training), so the comparison is honest.

Usage:
    python scripts/probes/droid_camera_readout.py \
        --cache_dir <droid_cache> --val_frac 0.2 \
        --ckpt on:<...>/v5.pt --ckpt blind:<...>/v5.pt \
        [--out_json <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.droid_window import DroidChunkPairs
from src.data.clevrer_window import chunk_collate
from src.model.camera_pose import CameraConditioner
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import LatentDecoder, SpatialGridDecoder


def build_from_ckpt(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    a = argparse.Namespace(**a) if isinstance(a, dict) else a
    d_pose = a.d_pose if getattr(a, "use_camera_pose", False) else 0
    enc = LatentEncoder3D(
        d_static=a.d_static, d_dyn=a.d_dyn, hidden_ch=a.enc_hidden_ch,
        shared_trunk=a.shared_trunk, pool_type=a.pool_type,
        n_queries=a.pool_queries, n_heads=a.pool_heads,
        static_grid=a.static_grid, d_pose=d_pose).to(device)
    enc.load_state_dict(ck["encoder"])
    enc.eval()
    # decoder — RE-APPLIES the pose the encoder inverted. The mean-pool baseline
    # uses LatentDecoder (flat z_static); the v5.8 spatial recipe uses
    # SpatialGridDecoder (z_static_grid + per-frame pose re-apply).
    if getattr(a, "decoder_type", "linear") == "spatial":
        dec = SpatialGridDecoder(d_static=a.d_static, static_grid=a.static_grid,
                                 d_dyn=a.d_dyn, hidden_ch=a.dec_hidden_ch,
                                 chunk_size_lat=getattr(a, "chunk_size_lat", 9),
                                 depth=getattr(a, "dec_depth", 3), d_pose=d_pose).to(device)
    else:
        dec = LatentDecoder(d_static=a.d_static, d_dyn=a.d_dyn,
                            hidden_ch=a.dec_hidden_ch,
                            chunk_size_lat=getattr(a, "chunk_size_lat", 9),
                            d_pose=d_pose).to(device)
    dec.load_state_dict(ck["decoder"])
    dec.eval()
    cc = None
    if getattr(a, "use_camera_pose", False):
        cc = CameraConditioner(pose_dim=a.pose_dim, d_pose=a.d_pose,
                               mode=a.pose_inject, n_trans=a.pose_n_trans,
                               mix_dim=a.enc_hidden_ch).to(device)
        cc.load_state_dict(ck["camera"])
        cc.eval()
    blind = bool(getattr(a, "cam_pose_blind", False))
    return enc, dec, cc, blind


@torch.no_grad()
def collect(enc, cc, blind, ds, device, batch_size=16):
    """Returns per-chunk arrays:
        zsta (N, D_s)            pooled static code
        zdn  (N, T)              per-frame ‖z_dyn‖
        zdyn (N, T, D_d)         per-frame z_dyn (for the regression readout)
        vmag (N, T)              per-frame ‖camera velocity‖
        vel  (N, T, 6)          per-frame camera velocity
        epid (N,)                episode id (for the within-episode readout)
    """
    zsta, zdn, zdyn, vmag, vel, epid = [], [], [], [], [], []
    for i0 in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(i0, min(i0 + batch_size, len(ds)))]
        b = chunk_collate(items)
        chunk = b["chunk_obs"].to(device)
        pemb = None
        if cc is not None:
            pemb = cc.embed(cc.relative(b["pose_obs"].to(device)))
            if blind:
                pemb = torch.zeros_like(pemb)
        o = enc(chunk, pose_emb=pemb) if cc is not None else enc(chunk)
        zd = o["z_dyn"]                       # (B, T, D_d)
        zsta.append(o["z_static"].cpu().numpy())
        zdn.append(zd.norm(dim=-1).cpu().numpy())
        zdyn.append(zd.cpu().numpy())
        v = b["vel_obs"].numpy()             # (B, T, 6)
        vel.append(v)
        vmag.append(np.linalg.norm(v, axis=-1))
        epid.append(b["video_id"].numpy())
    return (np.concatenate(zsta), np.concatenate(zdn), np.concatenate(zdyn),
            np.concatenate(vmag), np.concatenate(vel), np.concatenate(epid))


def pearson(x, y):
    x = x.ravel().astype(np.float64); y = y.ravel().astype(np.float64)
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-12 else 0.0


def linreg_r2(X, Y):
    """Multi-output least squares X->Y with bias; returns mean R² over outputs."""
    X = X.reshape(-1, X.shape[-1]).astype(np.float64)
    Y = Y.reshape(-1, Y.shape[-1]).astype(np.float64)
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    pred = Xb @ W
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
    return float(np.mean(r2))


def within_episode_ratio(zsta, epid):
    """Mean within-episode std of z_static / across-episode std. Lower = more
    viewpoint-stable. Uses only episodes with >=2 chunks."""
    per_dim_within = []
    for ep in np.unique(epid):
        m = epid == ep
        if m.sum() < 2:
            continue
        per_dim_within.append(zsta[m].std(axis=0))          # (D_s,)
    if not per_dim_within:
        return float("nan")
    within = np.mean(per_dim_within, axis=0)                 # (D_s,)
    across = zsta.std(axis=0)                                # (D_s,)
    return float(np.mean(within / np.clip(across, 1e-12, None)))


def psnr_pt(pred, gt):
    """pred,gt in [-1,1], any matching shape. Standard [-1,1]-image PSNR."""
    mse = ((pred.clamp(-1, 1) - gt.clamp(-1, 1)) / 2.0).pow(2).mean().item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def psnr_pass(enc, dec, cc, blind, ds, vae, vdt, device, batch_size=8, max_chunks=160):
    """Decode model reconstructions to pixels and compare to the true GT frames.
    Returns (model_psnr, vaefloor_psnr): model recon vs GT, and frozen-VAE
    round-trip vs GT (the reconstruction ceiling this model can reach)."""
    m_sum = m_n = f_sum = f_n = 0.0
    n_done = 0
    for i0 in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(i0, min(i0 + batch_size, len(ds)))]
        if "frames_obs" not in items[0]:
            return None, None                    # cache built without --save_frames
        b = chunk_collate(items)
        chunk = b["chunk_obs"].to(device)
        pemb = None
        if cc is not None:
            pemb = cc.embed(cc.relative(b["pose_obs"].to(device)))
            if blind:
                pemb = torch.zeros_like(pemb)
        o = enc(chunk, pose_emb=pemb) if cc is not None else enc(chunk)
        dec_cond = o["z_static_grid"] if isinstance(dec, SpatialGridDecoder) else o["z_static"]
        recon = dec(dec_cond, o["z_dyn"], pose_emb=pemb)         # (B,48,9,8,8)
        # GT frames -> (B,3,33,128,128) in [-1,1]
        gt = b["frames_obs"].to(device).float().div(127.5).sub(1.0)
        gt = gt.permute(0, 4, 1, 2, 3).contiguous()
        model_pix = vae.decode(recon.to(vdt)).sample.float()     # (B,3,33,128,128)
        floor_pix = vae.decode(chunk.to(vdt)).sample.float()
        m_sum += psnr_pt(model_pix, gt); m_n += 1
        f_sum += psnr_pt(floor_pix, gt); f_n += 1
        n_done += len(items)
        if n_done >= max_chunks:
            break
    return m_sum / max(m_n, 1), f_sum / max(f_n, 1)


def evaluate(tag, ckpt_path, ds, device, vae=None, vdt=None):
    enc, dec, cc, blind = build_from_ckpt(ckpt_path, device)
    zsta, zdn, zdyn, vmag, vel, epid = collect(enc, cc, blind, ds, device)
    row = {
        "tag": tag,
        "ckpt": ckpt_path,
        "has_camera": cc is not None,
        "blind": blind,
        "n_chunks": int(len(zsta)),
        "n_episodes": int(len(np.unique(epid))),
        # R1 — lower is better for pose-ON
        "corr_zdyn_camvel": pearson(zdn, vmag),
        "r2_zdyn_to_camvel": linreg_r2(zdyn, vel),
        # R2 — lower is better for pose-ON
        "zstatic_within_ep_ratio": within_episode_ratio(zsta, epid),
        # context
        "zdyn_norm_mean": float(zdn.mean()),
        "camvel_mag_mean": float(vmag.mean()),
    }
    if vae is not None:
        m_psnr, f_psnr = psnr_pass(enc, dec, cc, blind, ds, vae, vdt, device)
        row["psnr_model_vs_gt"] = m_psnr           # HIGHER better (moving-cam recon)
        row["psnr_vaefloor_vs_gt"] = f_psnr        # frozen-VAE ceiling (same for both)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL:PATH ; repeat for each arm (e.g. on:.../v5.pt).")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--psnr", action="store_true",
                    help="Also decode reconstructions through the frozen Wan VAE "
                         "and report pixel PSNR vs GT (needs --save_frames cache).")
    ap.add_argument("--vae_dtype", type=str, default="bfloat16")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Same split as training: read out on held-out episodes only.
    ds = DroidChunkPairs(args.cache_dir, split="val", val_frac=args.val_frac,
                         seed=args.seed, load_frames=args.psnr)
    print(f"[data] {len(ds)} val pairs from {args.cache_dir}")

    vae = vdt = None
    if args.psnr:
        from diffusers import AutoencoderKLWan
        vdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}[args.vae_dtype]
        print(f"[vae] loading frozen Wan VAE ({args.vae_dtype}) for pixel PSNR")
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=vdt)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        vae = vae.to(device)

    rows = []
    for spec in args.ckpt:
        tag, path = spec.split(":", 1)
        print(f"[eval] {tag}: {path}")
        rows.append(evaluate(tag, path, ds, device, vae=vae, vdt=vdt))

    # --- table ---
    cols = ["tag", "corr_zdyn_camvel", "r2_zdyn_to_camvel",
            "zstatic_within_ep_ratio", "zdyn_norm_mean", "n_chunks"]
    if args.psnr:
        cols[5:5] = ["psnr_model_vs_gt", "psnr_vaefloor_vs_gt"]
    w = {c: max(len(c), 10) for c in cols}
    print("\n" + "  ".join(c.rjust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(w[c]))
        print("  ".join(cells))

    print("\n[interpretation] pose conditioning WORKS if the pose-ON arm has "
          "LOWER corr_zdyn_camvel, LOWER r2_zdyn_to_camvel, and LOWER "
          "zstatic_within_ep_ratio than the BLIND control.")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
