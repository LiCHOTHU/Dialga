"""scripts/train_v5_libero.py — v5.1.2 chunk-wise trainer for LIBERO-90.

Forked from scripts/train_v5.py (CLEVRER) with these substitutions:

  * Data: LiberoChunkPairs replaces ClevrerChunkPairs. Splits by task_id +
    held-out 10 tasks for a downstream cross-task action probe.
  * L_attrs (CLEVRER color/material/shape CE) is replaced by
    L_clip_static — InfoNCE between z_static and the CLIP text embedding of
    the task prompt. Frees us from per-object labels we don't have.
  * EventHead's GT teacher is gripper-toggle-in-chunk (already encoded as
    gate_GT by the LIBERO dataset class). Architecture is unchanged.
  * No VAE unfreeze, no L_pixel for v1 — latent-space supervision only.

Same loss skeleton otherwise: L_recon + L_pred + L_fwd + L_consist (+ L_clip).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.libero_window import (LiberoChunkPairs, LiberoChunkPairsWithPixels,
                                    libero_collate)
from src.loss.info_nce import info_nce
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


# ---------- optional Wan VAE (only loaded if lambda_pixel > 0) ----------------

def _resolve_wan_path(name: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers") -> str:
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    repo_dir = Path(cache_root) / f"models--{name.replace('/', '--')}"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        snap = repo_dir / "snapshots" / ref_main.read_text().strip()
        if snap.exists(): return str(snap)
    return name


def load_wan_vae(dtype, device):
    """Frozen Wan-VAE for L_pixel. No unfreeze in v2 — pixel anchor only."""
    from diffusers import AutoencoderKLWan
    path = _resolve_wan_path()
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    return vae.to(device)


def _vae_decode(vae, latent):
    z = latent.to(next(vae.parameters()).dtype)
    out = vae.decode(z)
    pix = out.sample if hasattr(out, "sample") else out      # (B, 3, T, H, W)
    return pix.permute(0, 2, 1, 3, 4).contiguous().float()    # (B, T, 3, H, W)


# ---------- CLIP text embedding cache ----------------------------------------

def _resolve_clip_path(name: str = "openai/clip-vit-base-patch32") -> str:
    """Find the local CLIP snapshot dir if present; else return the repo name
    so HF Hub can fetch it. PACE compute nodes generally lack outbound HTTPS,
    so we strongly prefer the local snapshot."""
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    repo_dir = Path(cache_root) / f"models--{name.replace('/', '--')}"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        snap = repo_dir / "snapshots" / ref_main.read_text().strip()
        if snap.exists():
            return str(snap)
    return name


def _load_clip_text_encoder(device, dtype):
    """Load CLIP-ViT-B/32 text tower. We use the projected output (512-d).

    Returns (tokenizer, text_model). text_model.eval(); frozen.
    """
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    path = _resolve_clip_path()
    tok = CLIPTokenizer.from_pretrained(path)
    mdl = CLIPTextModelWithProjection.from_pretrained(path, torch_dtype=dtype).eval().to(device)
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl


@torch.no_grad()
def _build_prompt_embedding_lookup(prompts: dict[int, str], tokenizer, model,
                                   device) -> torch.Tensor:
    """Encode each unique task's prompt once. Returns (n_tasks, clip_dim) L2-normed.

    `prompts` maps task_id -> prompt string. We L2-normalize the projection so
    cosine = dot — matches the InfoNCE that consumes these vectors.
    """
    n = max(prompts.keys()) + 1
    rows = []
    for tid in range(n):
        text = prompts.get(tid, "")
        toks = tokenizer([text], padding=True, return_tensors="pt").to(device)
        out = model(**toks)
        e = out.text_embeds.float()                            # (1, 512)
        e = F.normalize(e, dim=-1)
        rows.append(e.squeeze(0))
    return torch.stack(rows, dim=0)                            # (n_tasks, 512)


# ---------- main loss path ---------------------------------------------------

def stage_at_epoch(ep: int, s1: int) -> int:
    """Two-stage schedule mirroring CLEVRER's: stage 1 = L_recon warmup,
    stage 2 = everything. (Stage 3 / events was dropped post-2026-05-20.)"""
    return 1 if ep <= s1 else 2


def compute_losses(batch, models, args, stage: int, device,
                   clip_lookup: torch.Tensor, clip_proj: torch.nn.Module,
                   vae=None):
    enc, dec, fwd, eh, ge, gp = models

    chunk_obs   = batch["chunk_obs"].to(device)
    chunk_pred  = batch["chunk_pred"].to(device)
    chunk_obs_b = batch["chunk_obs_b"].to(device)
    gate_GT     = batch["gate_GT"].to(device).float()             # (B,)
    task_id     = batch["task_id"].to(device)                     # (B,)

    enc_obs  = enc(chunk_obs)
    enc_pred = enc(chunk_pred)
    enc_b    = enc(chunk_obs_b)
    z_static_a = enc_obs["z_static"]
    z_dyn_obs  = enc_obs["z_dyn"]
    z_dyn_pred_target = enc_pred["z_dyn"].detach()
    z_static_b = enc_b["z_static"]

    z_dyn_last = z_dyn_obs[:, -1]
    T_chunk = z_dyn_obs.shape[1]

    z_dyn_pred_base = fwd.chunk_step(z_dyn_last, T_chunk)
    z_event = eh(z_dyn_last)
    correction = ge(z_event)
    correction_t = correction.unsqueeze(1).expand(-1, T_chunk, -1)
    z_dyn_pred_roll = z_dyn_pred_base + correction_t * gate_GT.unsqueeze(-1).unsqueeze(-1)

    recon_obs  = dec(z_static_a, z_dyn_obs)
    recon_pred = dec(z_static_a, z_dyn_pred_roll)

    L_recon = F.mse_loss(recon_obs, chunk_obs)
    L_pred  = F.mse_loss(recon_pred, chunk_pred)
    L_fwd   = F.mse_loss(z_dyn_pred_base, z_dyn_pred_target)

    if getattr(args, "consist_loss", "infonce") == "mse":
        L_infonce = F.mse_loss(z_static_a, z_static_b)
    else:
        L_infonce = info_nce(z_static_a, z_static_b,
                             temperature=args.infonce_temperature)

    # L_pixel — Wan-decode our recon, MSE vs the raw mp4 frames. The encoder
    # never saw a pixel-anchored signal in v1, which left the latents close to
    # Wan's in MSE but blurry under Wan's decoder (eval: 0.0215 vs CLEVRER e1
    # 0.0055). This is the anchor that fixes it.
    L_pixel = torch.zeros((), device=device)
    L_pixel_pred = torch.zeros((), device=device)
    pixel_on = (vae is not None and getattr(args, "lambda_pixel", 0.0) > 0.0
                and "pix_obs" in batch)
    if pixel_on:
        pix_obs_t  = batch["pix_obs"].to(device)
        pix_pred_t = batch["pix_pred"].to(device)
        pred_pix_obs  = _vae_decode(vae, recon_obs)
        pred_pix_pred = _vae_decode(vae, recon_pred)
        T_obs  = min(pred_pix_obs.shape[1],  pix_obs_t.shape[1])
        T_pred = min(pred_pix_pred.shape[1], pix_pred_t.shape[1])
        L_pixel      = F.mse_loss(pred_pix_obs [:, :T_obs ], pix_obs_t [:, :T_obs ])
        L_pixel_pred = F.mse_loss(pred_pix_pred[:, :T_pred], pix_pred_t[:, :T_pred])

    # L_clip — symmetric InfoNCE between projected z_static and prompt embed
    z_proj = clip_proj(z_static_a)                                 # (B, 512)
    z_proj = F.normalize(z_proj, dim=-1)
    text_e = clip_lookup[task_id]                                  # (B, 512), already L2-normed
    if args.lambda_clip > 0:
        L_clip = info_nce(z_proj, text_e, temperature=args.infonce_temperature)
    else:
        L_clip = torch.zeros((), device=device)

    # Event aux losses (logged, NOT in total — same as CLEVRER post-stage-3 removal)
    true_residual = z_dyn_pred_target - z_dyn_pred_base.detach()
    pred_residual_t = ge(eh(z_dyn_last)).unsqueeze(1).expand(-1, T_chunk, -1)
    L_event_aux = (F.mse_loss(pred_residual_t, true_residual, reduction="none")
                   * gate_GT.unsqueeze(-1).unsqueeze(-1)).mean()
    gate_logits = gp(z_dyn_last)
    L_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_GT)

    lambda_recon = getattr(args, "lambda_recon", 1.0)
    if stage == 1:
        total = lambda_recon * L_recon
        if pixel_on:
            total = total + args.lambda_pixel * L_pixel
    else:
        total = (lambda_recon * L_recon
                 + args.lambda_pred * L_pred
                 + args.lambda_fwd * L_fwd
                 + args.lambda_consist * L_infonce
                 + args.lambda_clip * L_clip)
        if pixel_on:
            total = total + args.lambda_pixel * (L_pixel + L_pixel_pred)
        if args.lambda_event_aux > 0:
            total = total + args.lambda_event_aux * L_event_aux
        if args.lambda_gate > 0:
            total = total + args.lambda_gate * L_gate

    with torch.no_grad():
        diag = {
            "z_static_norm":   z_static_a.norm(dim=-1).mean(),
            "z_static_std":    z_static_a.std(dim=0).mean(),
            "z_dyn_obs_norm":  z_dyn_obs.norm(dim=-1).mean(),
            "z_dyn_obs_std":   z_dyn_obs.std(dim=0).mean(),
            "z_dyn_roll_norm": z_dyn_pred_roll.norm(dim=-1).mean(),
            "z_event_norm":    z_event.norm(dim=-1).mean(),
            "gate_GT_mean":    gate_GT.mean(),
        }

    return {
        "recon": L_recon, "pred": L_pred, "fwd": L_fwd, "consist": L_infonce,
        "clip": L_clip, "event_aux": L_event_aux, "gate": L_gate,
        "pixel": L_pixel, "pixel_pred": L_pixel_pred,
        "total": total, **diag,
    }


# ---------- validation -------------------------------------------------------

@torch.no_grad()
def validate(models, val_loader, args, device, clip_lookup, clip_proj, vae=None):
    [m.eval() for m in models]
    clip_proj.eval()
    sums, n = {}, 0
    for batch in val_loader:
        out = compute_losses(batch, models, args, stage=2, device=device,
                             clip_lookup=clip_lookup, clip_proj=clip_proj, vae=vae)
        B = batch["chunk_obs"].shape[0]
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v) * B
        n += B
    [m.train() for m in models]
    clip_proj.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True,
                    help="Wan-VAE LIBERO cache directory (encode_libero_wan.py output).")
    ap.add_argument("--out_dir",   type=str, required=True)

    ap.add_argument("--epochs",        type=int, default=200)
    ap.add_argument("--stage1_epochs", type=int, default=5)
    ap.add_argument("--batch_size",    type=int, default=16)
    ap.add_argument("--num_workers",   type=int, default=4)
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-3)
    ap.add_argument("--lr_schedule",   type=str,   default="constant",
                    choices=["cosine", "constant"])

    ap.add_argument("--d_static", type=int, default=64)
    ap.add_argument("--d_dyn",    type=int, default=64)
    ap.add_argument("--d_state",  type=int, default=32)
    ap.add_argument("--d_event",  type=int, default=4)
    ap.add_argument("--enc_hidden_ch", type=int, default=128)
    ap.add_argument("--dec_hidden_ch", type=int, default=256)
    ap.add_argument("--chunk_size_lat", type=int, default=9)
    ap.add_argument("--clip_dim", type=int, default=512,
                    help="CLIP text projection dim. CLIP-ViT-B/32 = 512.")

    ap.add_argument("--lambda_recon",     type=float, default=1.0)
    ap.add_argument("--lambda_pred",      type=float, default=1.0)
    ap.add_argument("--lambda_fwd",       type=float, default=0.1)
    ap.add_argument("--lambda_consist",   type=float, default=1.0)
    ap.add_argument("--lambda_clip",      type=float, default=0.5,
                    help="LIBERO-specific: weight on InfoNCE between z_static "
                         "and CLIP-text prompt embedding. Replaces L_attrs.")
    ap.add_argument("--lambda_event_aux", type=float, default=0.0,
                    help="OFF by default — CLEVRER showed stage-3 events hurt "
                         "L_recon. Set >0 to re-enable with gripper-toggle teacher.")
    ap.add_argument("--lambda_gate",      type=float, default=0.0)
    ap.add_argument("--infonce_temperature", type=float, default=0.1)

    ap.add_argument("--val_every",  type=int, default=5)
    ap.add_argument("--n_holdout_tasks", type=int, default=10,
                    help="Number of tasks reserved for the cross-task action probe.")
    ap.add_argument("--task_holdout_seed", type=int, default=0)
    ap.add_argument("--val_every_k_episodes", type=int, default=10,
                    help="Within non-holdout tasks, episode_id %% k == 0 is val.")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--require_chunks_per_episode", type=int, default=3,
                    help="Episodes shorter than this many chunks are dropped. "
                         "Set 2 for tiny smoke caches; keep 3 in production.")
    ap.add_argument("--seed",       type=int, default=42)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every",  type=int, default=1)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--max_steps",  type=int, default=0)
    ap.add_argument("--early_stop_patience", type=int, default=0)
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    ap.add_argument("--consist_loss", type=str, default="infonce",
                    choices=["infonce", "mse"])
    ap.add_argument("--shared_trunk", action="store_true")
    ap.add_argument("--no_proj", action="store_true")
    ap.add_argument("--clip_dtype", type=str, default="float32",
                    choices=["float32", "float16", "bfloat16"])
    # Pixel-loss anchor (v2 fix). When >0 we load Wan-VAE and add
    # L_pixel = MSE(wan_dec(recon), raw_pix). Requires the mp4 dataset
    # via --libero_root.
    ap.add_argument("--lambda_pixel", type=float, default=0.0,
                    help="MSE on wan_decode(recon) vs raw frames. 0 = off "
                         "(v1 was off; v2 turns it on at 5.0 to fix blur).")
    ap.add_argument("--libero_root", type=str,
                    default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90_processed",
                    help="Path to mp4 root (needed when --lambda_pixel > 0).")
    ap.add_argument("--vae_model_id", type=str,
                    default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--vae_dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--wandb_project", type=str, default="dialga")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(args.wandb_project) and _WANDB_OK and args.wandb_mode != "disabled"
    if use_wandb:
        run_name = args.wandb_run_name or out_dir.name
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        os.environ.setdefault("WANDB_SILENT", "true")
        init_kw = dict(project=args.wandb_project, name=run_name,
                       mode=args.wandb_mode, config=vars(args), dir=str(out_dir))
        rid_file = out_dir / "wandb_run_id.txt"
        if args.resume and rid_file.exists():
            prev_id = rid_file.read_text().strip()
            if prev_id:
                init_kw.update(id=prev_id, resume="allow")
        wandb.init(**init_kw)
        try:
            rid_file.write_text(wandb.run.id)
        except Exception:
            pass
        print(f"[wandb] project={args.wandb_project} run={run_name}")
    else:
        print("[wandb] disabled")

    # ---- data ----
    use_pixels = args.lambda_pixel > 0.0
    ds_kw = dict(
        n_holdout_tasks=args.n_holdout_tasks,
        task_holdout_seed=args.task_holdout_seed,
        val_every_k=args.val_every_k_episodes,
        pair_seed=args.seed,
        max_episodes=args.max_episodes,
        require_chunks_per_video=args.require_chunks_per_episode,
    )
    if use_pixels:
        ds_train = LiberoChunkPairsWithPixels(args.cache_dir, args.libero_root,
                                              split="train", **ds_kw)
        ds_val   = LiberoChunkPairsWithPixels(args.cache_dir, args.libero_root,
                                              split="val",   **ds_kw)
    else:
        ds_train = LiberoChunkPairs(args.cache_dir, split="train", **ds_kw)
        ds_val   = LiberoChunkPairs(args.cache_dir, split="val",   **ds_kw)
    print(f"[data] train={len(ds_train)} pairs ({len(ds_train.kept_episodes)} eps), "
          f"val={len(ds_val)} pairs ({len(ds_val.kept_episodes)} eps), "
          f"holdout tasks={sorted(ds_train.holdout_task_ids)}")

    pin = (device.type == "cuda")
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=libero_collate,
                        pin_memory=pin, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=libero_collate,
                            pin_memory=pin)

    # ---- models ----
    enc = LatentEncoder3D(d_static=args.d_static, d_dyn=args.d_dyn,
                          hidden_ch=args.enc_hidden_ch,
                          shared_trunk=args.shared_trunk).to(device)
    dec = LatentDecoder(d_static=args.d_static, d_dyn=args.d_dyn,
                        hidden_ch=args.dec_hidden_ch,
                        chunk_size_lat=args.chunk_size_lat).to(device)
    fwd = ForwardDynamics(d_dyn=args.d_dyn, d_state=args.d_state,
                          no_proj=args.no_proj).to(device)
    eh = EventHead(d_dyn=args.d_dyn, d_event=args.d_event).to(device)
    ge = GEvent(d_event=args.d_event, d_dyn=args.d_dyn).to(device)
    gp = GatePredictor(d_dyn=args.d_dyn).to(device)
    clip_proj = torch.nn.Linear(args.d_static, args.clip_dim).to(device)
    models = (enc, dec, fwd, eh, ge, gp)

    # ---- CLIP text encoder + prompt-embedding lookup ----
    clip_dtype = {"float32": torch.float32, "float16": torch.float16,
                  "bfloat16": torch.bfloat16}[args.clip_dtype]
    print(f"[clip] loading openai/clip-vit-base-patch32 (dtype={args.clip_dtype})")
    clip_tok, clip_mdl = _load_clip_text_encoder(device, clip_dtype)

    # Build {task_id: prompt} from the dataset's task_id_map and any sample.
    task_prompt: dict[int, str] = {}
    # Walk the cache metadata via ds_train.windows + per-blob prompt — cheap.
    # We only need one prompt per task_id; use the first blob we see.
    for w in ds_train.windows:
        tid = int(w["task_id"])
        if tid in task_prompt: continue
        b = torch.load(ds_train.cache_dir / w["path"], map_location="cpu",
                       weights_only=False)
        task_prompt[int(b["task_id"])] = b["prompt"]
        if len(task_prompt) == len(ds_train.task_id_map):
            break
    # Held-out tasks won't appear in ds_train, but the lookup table is keyed
    # by all task_ids — fill from the cache directly for any missing ones.
    if len(task_prompt) < len(ds_train.task_id_map):
        missing = set(ds_train.task_id_map.values()) - set(task_prompt.keys())
        for w in ds_train.windows:
            tid = int(w["task_id"])
            if tid not in missing: continue
            b = torch.load(ds_train.cache_dir / w["path"], map_location="cpu",
                           weights_only=False)
            task_prompt[tid] = b["prompt"]
            missing.discard(tid)
            if not missing: break
    print(f"[clip] embedding {len(task_prompt)} unique task prompts")
    clip_lookup = _build_prompt_embedding_lookup(task_prompt, clip_tok, clip_mdl,
                                                 device).float()
    print(f"[clip] lookup shape {tuple(clip_lookup.shape)}")
    del clip_tok, clip_mdl  # free; we never re-encode

    n_total = sum(sum(p.numel() for p in m.parameters()) for m in models)
    n_total += sum(p.numel() for p in clip_proj.parameters())
    print(f"[model] enc {sum(p.numel() for p in enc.parameters())/1e3:.1f}K | "
          f"dec {sum(p.numel() for p in dec.parameters())/1e6:.2f}M | "
          f"fwd {sum(p.numel() for p in fwd.parameters())} | "
          f"clip_proj {sum(p.numel() for p in clip_proj.parameters())} | "
          f"total {n_total/1e6:.2f}M")

    # ---- Wan VAE (frozen) for the pixel anchor ----
    vae = None
    if use_pixels:
        vae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                     "float32": torch.float32}[args.vae_dtype]
        print(f"[vae] loading {args.vae_model_id} dtype={args.vae_dtype} (frozen)")
        vae = load_wan_vae(vae_dtype, device)
        n_vae = sum(p.numel() for p in vae.parameters())
        print(f"[vae] {n_vae/1e6:.1f}M params total, 0 trainable")

    params = []
    for m in models:
        params += list(m.parameters())
    params += list(clip_proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)

    history = []
    step = 0
    t0 = time.time()
    best_val = float("inf")
    best_epoch = 0
    val_no_improve = 0
    start_epoch = 1

    # ---- resume ----
    if args.resume:
        resume_path = (out_dir / "last.pt") if args.resume == "auto" else Path(args.resume)
        if resume_path.exists():
            print(f"[resume] loading {resume_path}")
            ck = torch.load(resume_path, map_location=device, weights_only=False)
            enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
            fwd.load_state_dict(ck["fwd"]); eh.load_state_dict(ck["event_head"])
            ge.load_state_dict(ck["g_event"]); gp.load_state_dict(ck["gate_predictor"])
            clip_proj.load_state_dict(ck["clip_proj"])
            if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
            if "scheduler" in ck: sched.load_state_dict(ck["scheduler"])
            step = int(ck.get("step", 0))
            best_val = float(ck.get("best_val", best_val))
            best_epoch = int(ck.get("best_epoch", best_epoch))
            val_no_improve = int(ck.get("val_no_improve", val_no_improve))
            if isinstance(ck.get("history"), list): history = ck["history"]
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[resume] continuing at epoch {start_epoch}/{args.epochs}")
        else:
            print(f"[resume] no checkpoint at {resume_path} — starting fresh")

    if start_epoch > args.epochs:
        (out_dir / "DONE").write_text(f"epochs={args.epochs}\n")
        if use_wandb: wandb.finish()
        return

    for ep in range(start_epoch, args.epochs + 1):
        stage = stage_at_epoch(ep, args.stage1_epochs)
        sums, n_batches = {}, 0
        [m.train() for m in models]
        clip_proj.train()
        for batch in loader:
            losses = compute_losses(batch, models, args, stage, device,
                                    clip_lookup=clip_lookup, clip_proj=clip_proj,
                                    vae=vae)
            total = losses["total"]
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n_batches += 1
            if args.max_steps > 0 and step >= args.max_steps: break
        sched.step()
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}

        val_metrics = None
        if ep % args.val_every == 0 or ep == args.epochs:
            val_metrics = validate(models, val_loader, args, device,
                                   clip_lookup=clip_lookup, clip_proj=clip_proj,
                                   vae=vae)

        if ep % args.log_every == 0 or ep == 1:
            keys = ["recon", "pred", "fwd", "consist", "clip", "event_aux", "gate"]
            if use_pixels: keys += ["pixel", "pixel_pred"]
            keys += ["total"]
            train_str = " ".join(f"{k}={avg[k]:.5f}" for k in keys if k in avg)
            diag_str = (f" |z_s_std={avg.get('z_static_std', 0):.3f}"
                        f" |z_d_norm={avg.get('z_dyn_obs_norm', 0):.3f}"
                        f" |gate={avg.get('gate_GT_mean', 0):.3f}")
            val_str = ""
            if val_metrics is not None:
                vp = f" val_pixel={val_metrics['pixel']:.5f}" if use_pixels else ""
                val_str = (f" | val_recon={val_metrics['recon']:.5f}"
                           f" val_fwd={val_metrics['fwd']:.5f}"
                           f" val_clip={val_metrics['clip']:.4f}{vp}")
            elapsed = time.time() - t0
            print(f"[ep {ep:3d}/{args.epochs} s{stage}] {train_str}{diag_str}{val_str} | "
                  f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        history.append({"epoch": ep, "stage": stage, "step": step,
                        "lr": opt.param_groups[0]["lr"],
                        "train": avg, "val": val_metrics})
        if use_wandb:
            log_row = {"epoch": ep, "stage": stage, "step": step,
                       "lr": opt.param_groups[0]["lr"], "wall_s": time.time() - t0,
                       **{f"train/{k}": v for k, v in avg.items()}}
            if val_metrics is not None:
                log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_row, step=ep)

        def _build_ckpt():
            ck = {
                "encoder": enc.state_dict(), "decoder": dec.state_dict(),
                "fwd": fwd.state_dict(), "event_head": eh.state_dict(),
                "g_event": ge.state_dict(), "gate_predictor": gp.state_dict(),
                "clip_proj": clip_proj.state_dict(),
                "args": vars(args), "epoch": ep, "step": step,
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "best_val": best_val, "best_epoch": best_epoch,
                "val_no_improve": val_no_improve, "history": history,
                "clip_lookup": clip_lookup,  # so the probe can reuse it
            }
            return ck

        ck = _build_ckpt()
        tmp_ckpt = out_dir / "last.pt.tmp"
        torch.save(ck, tmp_ckpt)
        os.replace(tmp_ckpt, out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0 or ep == args.epochs):
            torch.save(ck, out_dir / f"ckpt_ep{ep:04d}.pt")
            torch.save(ck, out_dir / "v5.pt")

        if val_metrics is not None and stage >= 2:
            cur = val_metrics["recon"]
            if cur < best_val - args.early_stop_min_delta:
                best_val = cur; best_epoch = ep; val_no_improve = 0
                torch.save(_build_ckpt(), out_dir / "v5_best.pt")
                print(f"  [best] val_recon={cur:.5f} -> saved v5_best.pt")
                if use_wandb:
                    wandb.log({"val/best_recon": best_val, "val/best_epoch": best_epoch},
                              step=ep)
            else:
                val_no_improve += 1
                if args.early_stop_patience > 0 and val_no_improve >= args.early_stop_patience:
                    print(f"[stop] early stop at ep {ep}; best val_recon={best_val:.5f} @ {best_epoch}")
                    break

        if args.max_steps > 0 and step >= args.max_steps: break

    torch.save(_build_ckpt(), out_dir / "v5.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "DONE").write_text(f"epoch={ep} best_val_recon={best_val:.5f}\n")
    print(f"\n[done] final: {avg}")
    print(f"[best] val_recon={best_val:.5f} at ep {best_epoch}")
    if use_wandb:
        wandb.run.summary["best_val"] = best_val
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()


if __name__ == "__main__":
    main()
