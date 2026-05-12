"""Evaluation-only script for the Wan-latent flow decoder.

Loads a trained decoder.pt, samples the 5 eval videos with configurable
solver / step count / classifier-free-guidance scale, and writes labelled
grids. Used for tuning inference quality without retraining.
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
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_DATASETS_CACHE", None)

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.clevrer_paired import ClevrerPairedDataset, paired_collate
from src.model.wan_flow_decoder import WanLatentFlowDecoder
from scripts.overfit_wan_flow import (
    build_encoder, load_wan_vae, latent_norm_buffers,
    encode_video, decode_latent, velocities_from_positions,
    save_grid,
)
from scripts.train_slot import encode_window


@torch.no_grad()
def euler_sample(decoder, n_steps, x0, q, v, z_cond, visib, z_i0_norm):
    x = x0
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)
    for i in range(n_steps):
        t_cur = ts[i].expand(x0.shape[0])
        dt = (ts[i + 1] - ts[i]).item()
        v_pred = decoder(x, t_cur, q, v, z_cond, visib, z_i0_norm)
        x = x + dt * v_pred
    return x


@torch.no_grad()
def heun_sample(decoder, n_steps, x0, q, v, z_cond, visib, z_i0_norm):
    """2nd-order Heun (predictor-corrector). Uses 2 NFE per step."""
    x = x0
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)
    for i in range(n_steps):
        t_cur = ts[i].expand(x0.shape[0])
        t_nxt = ts[i + 1].expand(x0.shape[0])
        dt = (ts[i + 1] - ts[i]).item()
        v1 = decoder(x, t_cur, q, v, z_cond, visib, z_i0_norm)
        x_pred = x + dt * v1
        v2 = decoder(x_pred, t_nxt, q, v, z_cond, visib, z_i0_norm)
        x = x + 0.5 * dt * (v1 + v2)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_decoder", required=True, help="path to decoder.pt")
    ap.add_argument("--ckpt_encoder", required=True, help="path to stage1.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps_list", default="16,32,64,128",
                    help="comma-separated step counts to evaluate")
    ap.add_argument("--solver", choices=["euler", "heun"], default="euler")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sample_fn = euler_sample if args.solver == "euler" else heun_sample

    # load decoder
    print("Loading decoder:", args.ckpt_decoder)
    dec_ckpt = torch.load(args.ckpt_decoder, map_location=device, weights_only=False)
    train_args = dec_ckpt["args"]
    cfg = dec_ckpt["config"]
    print("Decoder train args:", {k: train_args[k] for k in
                                  ["cond_source", "use_gt_pos", "use_i0", "d_model", "n_blocks"]})

    # encoder + dataset
    enc_ckpt = torch.load(args.ckpt_encoder, map_location=device, weights_only=False)
    pos_norm = float(cfg["dataset"]["pos_normalize"])
    d_static = int(cfg["model"].get("d_static", 16))

    dataset = ClevrerPairedDataset(
        data_dir=str(cfg["dataset"]["data_dir"]),
        annotation_dir=str(cfg["dataset"]["annotation_dir"]),
        split=str(cfg["dataset"]["split"]),
        window_length=int(cfg["training"]["window_length"]),
        frames_per_video=int(cfg["dataset"]["video_num_frames"]),
        windows_per_video=int(cfg["training"]["windows_per_video"]),
        max_videos=int(cfg["training"]["max_videos"]),
        max_objects=int(cfg["dataset"]["max_objects"]),
        coordinate_mode=str(cfg["dataset"]["coordinate_mode"]),
        image_size=int(cfg["dataset"]["image_size"]),
        seed=int(cfg["training"]["seed"]),
    )
    attr_dim = dataset.attr_dim

    encoder = build_encoder(enc_ckpt, attr_dim, d_static, device)
    vae = load_wan_vae(args.model_id, dtype, device)
    lat_mean, lat_std = latent_norm_buffers(vae, device, torch.float32)

    cond_feat_dim = attr_dim if train_args["cond_source"] == "gt_attrs" else d_static
    decoder = WanLatentFlowDecoder(
        latent_channels=48, latent_grid=8, latent_T=2,
        d_model=int(train_args.get("d_model", 384)),
        n_heads=int(train_args.get("n_heads", 6)),
        n_blocks=int(train_args.get("n_blocks", 6)),
        t_dim=int(train_args.get("d_model", 384)),
        d_static=cond_feat_dim,
        max_objects=int(cfg["dataset"]["max_objects"]),
        window_length=int(cfg["training"]["window_length"]),
        use_i0=bool(train_args.get("use_i0", True)),
    ).to(device)
    state_key = "ema_decoder_state_dict" if "ema_decoder_state_dict" in dec_ckpt else "decoder_state_dict"
    decoder.load_state_dict(dec_ckpt[state_key])
    decoder.eval()
    print(f"Loaded weights from key '{state_key}'")

    # eval batch — same selection as training script
    seen = {}
    for i in range(len(dataset)):
        s = dataset[i]
        if s["video_id"] not in seen and s["start_frame"] == 0:
            seen[s["video_id"]] = s
        if len(seen) >= 5:
            break
    if len(seen) == 0:
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
    N, T = frames.shape[:2]

    with torch.no_grad():
        z = encode_video(vae, frames).float()
        ceiling = decode_latent(vae, z.to(dtype)).float()

        if train_args.get("use_i0", True):
            z_i0 = encode_video(vae, frames[:, :1]).float()
            z_i0_norm = (z_i0 - lat_mean) / lat_std
        else:
            z_i0_norm = None

        if train_args["cond_source"] == "gt_attrs":
            z_cond = attrs
            q = gt_pos if train_args["use_gt_pos"] else encode_window(encoder, frames.float(), attrs)
            if isinstance(q, tuple): q = q[0]
        else:
            enc_out = encode_window(encoder, frames.float(), attrs)
            q_enc = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            zs = enc_out[1] if isinstance(enc_out, tuple) and len(enc_out) > 1 else None
            if zs is not None:
                w = visib.unsqueeze(-1)
                z_cond = (zs * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
            else:
                z_cond = attrs
            q = gt_pos if train_args["use_gt_pos"] else q_enc
        v = velocities_from_positions(q)

    frames_f = frames.float()
    T_out = ceiling.shape[1]
    T_cmp = min(T, T_out)
    ceil_mse = (ceiling[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()
    print(f"\nWan-VAE ceiling MSE: {ceil_mse:.6f}")

    g = torch.Generator(device=device).manual_seed(args.seed)
    x0_seed = torch.randn(N, 48, 2, 8, 8, generator=g, device=device)

    step_results = []
    for n_steps in [int(s) for s in args.steps_list.split(",")]:
        with torch.no_grad():
            x = sample_fn(decoder, n_steps, x0_seed.clone(), q, v, z_cond, visib, z_i0_norm)
            z_sample = x * lat_std + lat_mean
            rec = decode_latent(vae, z_sample.to(dtype)).float()
        rec_mse = (rec[:, :T_cmp] - frames_f[:, :T_cmp]).pow(2).mean().item()
        step_results.append((n_steps, rec_mse, rec))
        print(f"[{args.solver:5s} steps={n_steps:4d}] pixel MSE {rec_mse:.6f} "
              f"({rec_mse/ceil_mse:.2f}x ceiling)")

    # Save per-video grids: GT / ceiling / each step count
    for i, s in enumerate(samples):
        rows = []
        rows.extend([frames_f[i, t] for t in range(T_cmp)])
        rows.extend([ceiling[i, t] for t in range(T_cmp)])
        for n_steps, _, rec in step_results:
            rows.extend([rec[i, t] for t in range(T_cmp)])
        save_grid(rows, out_dir / f"video_{s['video_id']}_grid.png", nrow=T_cmp)
    labels = ["GT", "Wan-VAE ceiling"] + [f"{args.solver} {n}" for n, _, _ in step_results]
    print(f"Grids saved under {out_dir}/  rows: {' / '.join(labels)}")


if __name__ == "__main__":
    main()
