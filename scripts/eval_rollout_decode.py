"""Rung 6: rollout-then-decode probe.

For each of the 5 overfit CLEVRER videos, decode three q-variants through the
trained Wan-flow decoder and compare pixel MSE against ground-truth video:

  variant A — GT positions across all W frames (control: best-case)
  variant B — encoder predictions across all W frames (rung 3 baseline)
  variant C — encoder for t=0,1; AccelNet/Lagrangian rolls forward to t=2..W-1
              (rung 6: the central project claim)

Reports per-video MSE for each variant, plus per-frame drift profile and
labelled grids.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HF_CACHE = os.path.expanduser("~/.cache/huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ["HF_HOME"] = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_HF_CACHE, "hub")

import torch
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.accel_net import AccelNet
from src.model.slot_lagrangian import CollisionImpulse
from src.model.wan_flow_decoder import WanLatentFlowDecoder
from scripts.overfit_wan_flow import (
    build_encoder, load_wan_vae, latent_norm_buffers,
    encode_video, decode_latent, velocities_from_positions, save_grid,
)
from scripts.train_slot import (
    _autoregressive_rollout_eval, encode_window, pool_z_static,
)


@torch.no_grad()
def euler_sample(decoder, n_steps, x0, q, v, z_cond, visib, z_i0_norm):
    x = x0
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)
    for i in range(n_steps):
        t_cur = ts[i].expand(x0.shape[0])
        dt = (ts[i + 1] - ts[i]).item()
        x = x + dt * decoder(x, t_cur, q, v, z_cond, visib, z_i0_norm)
    return x


def build_dynamics(cfg, attr_dim, device):
    d_static = int(cfg.model.get("d_static", 16))
    m = AccelNet(
        num_state_dims=int(cfg.model.num_state_dims),
        attr_dim=d_static,
        hidden=int(cfg.model.lagrangian_hidden),
        use_pair=bool(cfg.model.lagrangian_use_pair),
    )
    return m.to(device).eval(), "accel"


def build_impulse(cfg, device):
    d_static = int(cfg.model.get("d_static", 16))
    return CollisionImpulse(
        attr_dim=d_static,
        hidden=int(cfg.model.get("impulse_hidden", 64)),
    ).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_slot", required=True, help="path to slot-pipeline stage2.pt")
    ap.add_argument("--ckpt_decoder", required=True, help="path to wan-flow decoder.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=8,
                    help="Euler steps for flow sampling (8 was best for v2_big)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load slot pipeline (encoder + dynamics + impulse)
    slot_ckpt = torch.load(args.ckpt_slot, map_location=device, weights_only=False)
    cfg = OmegaConf.create(slot_ckpt["config"])
    pos_norm = float(cfg.dataset.pos_normalize)

    dataset = ClevrerPairedDataset(
        data_dir=str(cfg.dataset.data_dir),
        annotation_dir=str(cfg.dataset.annotation_dir),
        split=str(cfg.dataset.split),
        window_length=int(cfg.training.window_length),
        frames_per_video=int(cfg.dataset.video_num_frames),
        windows_per_video=int(cfg.training.windows_per_video),
        max_videos=int(cfg.training.max_videos),
        max_objects=int(cfg.dataset.max_objects),
        coordinate_mode=str(cfg.dataset.coordinate_mode),
        image_size=int(cfg.dataset.image_size),
        seed=int(cfg.training.seed),
    )
    attr_dim = dataset.attr_dim
    d_static = int(cfg.model.get("d_static", 16))

    encoder = build_encoder(slot_ckpt, attr_dim, d_static, device)
    dyn, dyn_type = build_dynamics(cfg, attr_dim, device)
    dyn.load_state_dict(slot_ckpt["lagrangian_state_dict"])
    impulse = build_impulse(cfg, device)
    if "impulse_state_dict" in slot_ckpt:
        impulse.load_state_dict(slot_ckpt["impulse_state_dict"])
    print(f"Slot pipeline: encoder={cfg.model.encoder_type} dyn={dyn_type} d_static={d_static}")

    # ---- Load Wan VAE + flow decoder
    vae = load_wan_vae(args.model_id, dtype, device)
    lat_mean, lat_std = latent_norm_buffers(vae, device, torch.float32)

    dec_ckpt = torch.load(args.ckpt_decoder, map_location=device, weights_only=False)
    train_args = dec_ckpt["args"]
    decoder = WanLatentFlowDecoder(
        latent_channels=48, latent_grid=8, latent_T=2,
        d_model=int(train_args.get("d_model", 384)),
        n_heads=int(train_args.get("n_heads", 6)),
        n_blocks=int(train_args.get("n_blocks", 6)),
        t_dim=int(train_args.get("d_model", 384)),
        d_static=(attr_dim if train_args["cond_source"] == "gt_attrs" else d_static),
        max_objects=int(cfg.dataset.max_objects),
        window_length=int(cfg.training.window_length),
        use_i0=bool(train_args.get("use_i0", True)),
    ).to(device)
    state_key = "ema_decoder_state_dict" if "ema_decoder_state_dict" in dec_ckpt else "decoder_state_dict"
    decoder.load_state_dict(dec_ckpt[state_key])
    decoder.eval()
    print(f"Wan-flow decoder: cond_source={train_args['cond_source']} "
          f"use_i0={train_args.get('use_i0', True)} use_gt_pos_train={train_args.get('use_gt_pos', False)} "
          f"loaded key={state_key}")

    # ---- 5-video batch
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= 5:
            break
    if not seen:
        for i in range(len(dataset)):
            s = dataset[i]
            if s["video_id"] not in seen:
                seen[s["video_id"]] = s
            if len(seen) >= 5:
                break
    samples = list(seen.values())
    batch = paired_collate(samples)

    frames = batch["frames"].to(device).to(dtype)
    gt_pos = batch["positions"].to(device).float() / pos_norm
    visib = batch["visibility"].to(device).float()
    attrs = batch["attrs"].to(device).float()
    collisions = batch["collisions"]
    N, T = frames.shape[:2]
    print(f"\n5-video batch: N={N}, T={T}, max_objects={gt_pos.shape[2]}")

    # ---- Encode + pool z_static (all frames)
    with torch.no_grad():
        q_enc, z_static_pf = encode_window(encoder, frames.float(), attrs)
        z_static = pool_z_static(z_static_pf, visib)             # (N, K, d_static)

    # ---- Build the three q-variants in normalized scale
    # variant A: GT q (control)
    q_A = gt_pos.clone()
    # variant B: encoder q (rung 3)
    q_B = q_enc.clone()
    # variant C: rollout q — use encoder for t=0,1 then dynamics
    #            (matches "predict from observed history" framing)
    q_seed = q_enc.clone()
    q_C = _autoregressive_rollout_eval(
        dyn, impulse,
        q_seed, visib, z_static,
        collisions_per_sample=collisions,
        solver_alpha=float(cfg.training.solver_alpha),
        solver_steps=max(int(cfg.training.solver_steps), 2),
        direction_clip=0.5,
        dynamics_type=dyn_type,
    )

    # ---- Wan-VAE encode (for ceiling reference) and i0 conditioning
    with torch.no_grad():
        z_full = encode_video(vae, frames).float()
        ceiling = decode_latent(vae, z_full.to(dtype)).float()

        if bool(train_args.get("use_i0", True)):
            z_i0 = encode_video(vae, frames[:, :1]).float()
            z_i0_norm = (z_i0 - lat_mean) / lat_std
        else:
            z_i0_norm = None

    cond_source = train_args["cond_source"]
    if cond_source == "gt_attrs":
        z_cond = attrs
    else:
        z_cond = z_static

    # ---- Sample latents from the flow decoder for each q-variant
    g = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.randn(N, 48, 2, 8, 8, generator=g, device=device)

    def sample_and_decode(q):
        v = velocities_from_positions(q)
        with torch.no_grad():
            x = euler_sample(decoder, args.num_steps, x0.clone(),
                             q, v, z_cond, visib, z_i0_norm)
            z_sample = x * lat_std + lat_mean
            rec = decode_latent(vae, z_sample.to(dtype)).float()
        return rec

    rec_A = sample_and_decode(q_A)
    rec_B = sample_and_decode(q_B)
    rec_C = sample_and_decode(q_C)

    frames_f = frames.float()
    T_out = ceiling.shape[1]
    T_cmp = min(T, T_out)
    ceil_mse = (ceiling[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()
    print(f"\nWan-VAE ceiling MSE: {ceil_mse:.6f}")

    # ---- Per-video MSE for each variant
    print(f"\nPer-video pixel MSE  (steps={args.num_steps}, ceiling={ceil_mse:.6f}):\n")
    print(f"{'Video':>6} | {'A: GT q':>10} | {'B: enc q':>10} | {'C: roll q':>10} "
          f"| {'A/ceil':>7} | {'B/ceil':>7} | {'C/ceil':>7}")
    print("-" * 88)

    rows_for_grid = []
    per_video = []
    for n, s in enumerate(samples):
        m_A = (rec_A[n, :T_cmp] - frames_f[n, :T_cmp]).pow(2).mean().item()
        m_B = (rec_B[n, :T_cmp] - frames_f[n, :T_cmp]).pow(2).mean().item()
        m_C = (rec_C[n, :T_cmp] - frames_f[n, :T_cmp]).pow(2).mean().item()
        per_video.append((s["video_id"], m_A, m_B, m_C))
        print(f"{s['video_id']:>6} | {m_A:>10.6f} | {m_B:>10.6f} | {m_C:>10.6f} "
              f"| {m_A/ceil_mse:>7.2f} | {m_B/ceil_mse:>7.2f} | {m_C/ceil_mse:>7.2f}")

    a_mean = sum(r[1] for r in per_video) / len(per_video)
    b_mean = sum(r[2] for r in per_video) / len(per_video)
    c_mean = sum(r[3] for r in per_video) / len(per_video)
    print("-" * 88)
    print(f"{'mean':>6} | {a_mean:>10.6f} | {b_mean:>10.6f} | {c_mean:>10.6f} "
          f"| {a_mean/ceil_mse:>7.2f} | {b_mean/ceil_mse:>7.2f} | {c_mean/ceil_mse:>7.2f}")

    # ---- Per-frame drift profile (rollout variant) for video 0
    print(f"\nPer-frame pixel MSE (variant C, video {samples[0]['video_id']}):")
    n = 0
    for t in range(T_cmp):
        mse_t = (rec_C[n, t] - frames_f[n, t]).pow(2).mean().item()
        marker = " (encoder seed)" if t < 2 else ""
        print(f"  t={t}: pixel MSE = {mse_t:.6f}  ({mse_t/ceil_mse:.2f}× ceiling){marker}")

    # ---- Save 5-row grids per video: GT / ceiling / A / B / C
    for n, s in enumerate(samples):
        rows = []
        rows.extend([frames_f[n, t] for t in range(T_cmp)])
        rows.extend([ceiling[n, t] for t in range(T_cmp)])
        rows.extend([rec_A[n, t] for t in range(T_cmp)])
        rows.extend([rec_B[n, t] for t in range(T_cmp)])
        rows.extend([rec_C[n, t] for t in range(T_cmp)])
        save_grid(rows, out_dir / f"video_{s['video_id']}_grid.png", nrow=T_cmp)
    print(f"\nGrids saved under {out_dir}/  rows: GT / ceiling / GT-q / enc-q / rollout-q")


if __name__ == "__main__":
    main()
