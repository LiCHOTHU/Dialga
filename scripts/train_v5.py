"""scripts/train_v5.py — v5.1 chunk-wise trainer with stage-gated losses.

Stages (each is a contiguous range of epochs):

  Stage 1: L_recon only.                  Warm up the autoencoder.
  Stage 2: + L_pred + L_fwd + L_infonce.  Predictive pressure + contrastive
                                          identity pressure on z_static.
  Stage 3: + L_event_aux + L_gate.        Adds the supervised event channel.

Each training sample is a triplet of chunks from one video:
  chunk_obs    — encoded as the "observed" half
  chunk_pred   — predicted by rolling z_dyn forward via ForwardDynamics
  chunk_obs_b  — InfoNCE positive (different chunk, same video)
plus a scalar gate_GT (collision occurred in chunk_pred).
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

from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.loss.info_nce import info_nce
from src.model.attrs_head import AttrsHead
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D

N_COLOR    = len(COLOR_VOCAB)
N_MATERIAL = len(MATERIAL_VOCAB)
N_SHAPE    = len(SHAPE_VOCAB)


def _modal_labels(attrs: torch.Tensor, slot_mask: torch.Tensor,
                  lo: int, hi: int) -> torch.Tensor:
    """Modal class index per video for attribute slice [lo:hi).

    attrs     : (B, K, A) one-hot blocks
    slot_mask : (B, K)    bool
    Returns   : (B,) long; -100 (CE ignore_index) where no real slots.
    """
    block = attrs[..., lo:hi]                              # (B, K, C)
    cls = block.argmax(dim=-1)                             # (B, K)
    n_class = hi - lo
    onehot = F.one_hot(cls, n_class).float()               # (B, K, C)
    masked = onehot * slot_mask.float().unsqueeze(-1)
    counts = masked.sum(dim=1)                             # (B, C)
    labels = counts.argmax(dim=-1)                         # (B,)
    valid = slot_mask.bool().any(dim=-1)                   # (B,)
    return torch.where(valid, labels, torch.full_like(labels, -100))


def stage_at_epoch(ep: int, s1: int, s2: int) -> int:
    """Two-stage training only. Stage 3 (events) was removed after the
    2026-05-20 120-epoch run showed it actively hurt val_recon (0.019 -> 0.021)
    when L_event_aux + L_gate gradients backed into the encoder. EventHead /
    GEvent / GatePredictor still instantiate but receive no gradient.
    """
    if ep <= s1: return 1
    return 2


# --------------------------------------------------------------------- losses

def compute_losses(batch, models, args, stage: int, device):
    """Compute ALL six losses every step (logged regardless of stage).
    Stage-gating controls only which losses sum into `total`.

    Wiring (per v5.1 spec):
      * One encoder pass on each of chunk_obs / chunk_pred / chunk_obs_b.
      * z_static_a reused for both recon_obs and recon_pred (identity-reuse).
      * Event correction is injected into the rollout from stage 2 onward;
        GEvent is zero-init so stage 2 is naturally a no-op until L_event_aux
        starts moving g_event in stage 3.
      * L_fwd uses the BASE step (no event correction).
      * L_event_aux trains only event_head + g_event (both fwd and target
        encoded z_dyn are detached).
    """
    enc, dec, fwd, eh, ge, gp, ah = models

    chunk_obs   = batch["chunk_obs"].to(device)       # (B, C, T, H, W)
    chunk_pred  = batch["chunk_pred"].to(device)
    chunk_obs_b = batch["chunk_obs_b"].to(device)
    gate_GT     = batch["gate_GT"].to(device).float() # (B,) in {0, 1}
    attrs       = batch["attrs"].to(device)            # (B, K, A) one-hot
    slot_mask   = batch["slot_mask"].to(device)        # (B, K) bool

    # ---- three encoder passes ----
    enc_obs  = enc(chunk_obs)
    enc_pred = enc(chunk_pred)
    enc_b    = enc(chunk_obs_b)
    z_static_a = enc_obs["z_static"]                  # (B, D_s)
    z_dyn_obs  = enc_obs["z_dyn"]                     # (B, T, D_d)  — per-frame
    z_dyn_pred_target = enc_pred["z_dyn"].detach()    # (B, T, D_d), detached
    z_static_b = enc_b["z_static"]

    # ---- last observed frame's z_dyn drives rollout/event prediction ----
    z_dyn_last = z_dyn_obs[:, -1]                     # (B, D_d)
    T_chunk = z_dyn_obs.shape[1]

    # ---- rollout: ONE chunk-to-chunk step, expands to T per-frame outputs ----
    z_dyn_pred_base = fwd.chunk_step(z_dyn_last, T_chunk)     # (B, T, D_d)
    z_event = eh(z_dyn_last)                                  # (B, D_e)
    correction = ge(z_event)                                  # (B, D_d) — zero at init
    # broadcast correction to all predicted frames, scaled by gate
    correction_t = correction.unsqueeze(1).expand(-1, T_chunk, -1)
    z_dyn_pred_roll = z_dyn_pred_base + correction_t * gate_GT.unsqueeze(-1).unsqueeze(-1)

    # ---- reconstructions (z_static_a reused for both) ----
    recon_obs = dec(z_static_a, z_dyn_obs)
    recon_pred = dec(z_static_a, z_dyn_pred_roll)

    # ---- six losses, always computed ----
    L_recon = F.mse_loss(recon_obs, chunk_obs)
    L_pred = F.mse_loss(recon_pred, chunk_pred)
    L_fwd = F.mse_loss(z_dyn_pred_base, z_dyn_pred_target)
    # Ablation 1: --consist_loss mse replaces InfoNCE with plain MSE on z_static.
    if getattr(args, "consist_loss", "infonce") == "mse":
        L_infonce = F.mse_loss(z_static_a, z_static_b)
    else:
        L_infonce = info_nce(z_static_a, z_static_b,
                             temperature=args.infonce_temperature)

    # L_event_aux: at gate=1, g_event(event_head(z_dyn_last)) (broadcast across T)
    # must match (z_dyn_pred_target - z_dyn_pred_base). Both sides on the target
    # detached so EventHead+GEvent are the only modules updated by this term.
    true_residual = z_dyn_pred_target - z_dyn_pred_base.detach()
    pred_residual_t = ge(eh(z_dyn_last)).unsqueeze(1).expand(-1, T_chunk, -1)
    L_event_aux = (F.mse_loss(pred_residual_t, true_residual, reduction="none")
                   * gate_GT.unsqueeze(-1).unsqueeze(-1)).mean()

    gate_logits = gp(z_dyn_last)
    L_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_GT)

    # ---- L_attrs (Exp 1): supervised CE on z_static for color/material/shape ----
    # Modal label per video (most-common class across visible slots); CE
    # ignores videos with no real slots via ignore_index=-100.
    attrs_logits = ah(z_static_a)
    y_color    = _modal_labels(attrs, slot_mask, 0,                          N_COLOR)
    y_material = _modal_labels(attrs, slot_mask, N_COLOR,                    N_COLOR + N_MATERIAL)
    y_shape    = _modal_labels(attrs, slot_mask, N_COLOR + N_MATERIAL,       N_COLOR + N_MATERIAL + N_SHAPE)
    L_attrs = (F.cross_entropy(attrs_logits["color"],    y_color,    ignore_index=-100)
               + F.cross_entropy(attrs_logits["material"], y_material, ignore_index=-100)
               + F.cross_entropy(attrs_logits["shape"],    y_shape,    ignore_index=-100))

    # ---- stage gating: which contribute to total ----
    # Stage 3 (events) was removed after it regressed val_recon. L_event_aux and
    # L_gate are still computed (for logging) but never enter `total`. EventHead /
    # GEvent / GatePredictor accordingly receive no gradient and stay near init.
    # L_attrs joins from stage 2 with weight λ_attrs (Exp 1: 0.05).
    if stage == 1:
        total = L_recon
    else:  # stage 2 — production loss for the rest of training
        total = (L_recon
                 + args.lambda_pred * L_pred
                 + args.lambda_fwd * L_fwd
                 + args.lambda_consist * L_infonce
                 + args.lambda_attrs * L_attrs)

    # ---- diagnostics (always logged, never contribute to loss) ----
    with torch.no_grad():
        diag = {
            "z_static_norm":  z_static_a.norm(dim=-1).mean(),
            "z_static_std":   z_static_a.std(dim=0).mean(),
            "z_dyn_obs_norm": z_dyn_obs.norm(dim=-1).mean(),    # avg over (B,T)
            "z_dyn_obs_std":  z_dyn_obs.std(dim=0).mean(),
            "z_dyn_roll_norm": z_dyn_pred_roll.norm(dim=-1).mean(),
            "z_event_norm":   z_event.norm(dim=-1).mean(),
            "gate_GT_mean":   gate_GT.mean(),
        }

    return {
        "recon": L_recon, "pred": L_pred, "fwd": L_fwd, "consist": L_infonce,
        "event_aux": L_event_aux, "gate": L_gate, "attrs": L_attrs, "total": total,
        **diag,
    }


# ----------------------------------------------------------------- validation

@torch.no_grad()
def validate(models, val_loader, args, device):
    [m.eval() for m in models]
    sums = {}
    n = 0
    for batch in val_loader:
        # stage=3 so every loss + every diagnostic is computed
        out = compute_losses(batch, models, args, stage=3, device=device)
        B = batch["chunk_obs"].shape[0]
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v) * B
        n += B
    [m.train() for m in models]
    return {k: v / max(n, 1) for k, v in sums.items()}


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--out_dir",   type=str, required=True)
    ap.add_argument("--epochs",        type=int, default=24)
    ap.add_argument("--stage1_epochs", type=int, default=3)
    ap.add_argument("--stage2_epochs", type=int, default=16)
    ap.add_argument("--batch_size",    type=int, default=16)
    ap.add_argument("--num_workers",   type=int, default=0)
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-3)
    ap.add_argument("--lr_schedule",   type=str,   default="cosine",
                    choices=["cosine", "constant"])

    ap.add_argument("--d_static", type=int, default=32)
    ap.add_argument("--d_dyn",    type=int, default=16)
    ap.add_argument("--d_state",  type=int, default=8)
    ap.add_argument("--d_event",  type=int, default=4)
    ap.add_argument("--enc_hidden_ch", type=int, default=32)
    ap.add_argument("--dec_hidden_ch", type=int, default=64)
    ap.add_argument("--chunk_size_lat", type=int, default=9)

    ap.add_argument("--lambda_pred",      type=float, default=1.0)
    ap.add_argument("--lambda_fwd",       type=float, default=0.1)
    ap.add_argument("--lambda_consist",   type=float, default=1.0)
    ap.add_argument("--lambda_event_aux", type=float, default=0.1)
    ap.add_argument("--lambda_gate",      type=float, default=0.1)
    ap.add_argument("--lambda_attrs",     type=float, default=0.0,
                    help="Exp 1: weight on supervised CE (color+material+shape) "
                         "from z_static via AttrsHead. 0.0 disables.")
    ap.add_argument("--attrs_hidden",     type=int,   default=0,
                    help="AttrsHead trunk width; 0 = linear classifier.")
    ap.add_argument("--infonce_temperature", type=float, default=0.1)

    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--val_every",  type=int,   default=2)
    ap.add_argument("--max_videos", type=int,   default=0)
    ap.add_argument("--seed",       type=int,   default=42)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every",  type=int, default=1)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--max_steps",  type=int, default=0)
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="Stop when val_recon hasn't improved for this many val checks. "
                         "0 disables. Requires val_loader. Best ckpt saved as v5_best.pt.")
    ap.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                    help="Minimum absolute improvement on val_recon to count as progress.")
    ap.add_argument("--consist_loss", type=str, default="infonce",
                    choices=["infonce", "mse"],
                    help="Ablation 1: 'mse' uses the original v5 MSE consistency "
                         "instead of contrastive InfoNCE on z_static.")
    ap.add_argument("--shared_trunk", action="store_true",
                    help="Ablation 2: single Conv3d trunk feeds both heads "
                         "(disables independent static/dyn trunks).")
    ap.add_argument("--no_proj", action="store_true",
                    help="Exp 3: drop the W_proj/W_unproj bottleneck in "
                         "ForwardDynamics. Requires d_state == d_dyn.")
    ap.add_argument("--wandb_project", type=str, default="dialga",
                    help="W&B project name. Set to '' to disable W&B.")
    ap.add_argument("--wandb_run_name", type=str, default=None,
                    help="W&B run name. Defaults to basename of --out_dir.")
    ap.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb init ----
    use_wandb = bool(args.wandb_project) and _WANDB_OK and args.wandb_mode != "disabled"
    if use_wandb:
        run_name = args.wandb_run_name or out_dir.name
        # Quiet wandb's own dir spam (matplotlib etc.)
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        os.environ.setdefault("WANDB_SILENT", "true")
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(out_dir),
        )
        print(f"[wandb] project={args.wandb_project} run={run_name} mode={args.wandb_mode}")
    else:
        print("[wandb] disabled" + (" (not installed)" if not _WANDB_OK else ""))

    # ---- data ----
    if args.val_frac > 0:
        ds_train = ClevrerChunkPairs(args.cache_dir, split="train",
                                     val_frac=args.val_frac, seed=args.seed,
                                     max_videos=args.max_videos)
        ds_val = ClevrerChunkPairs(args.cache_dir, split="val",
                                   val_frac=args.val_frac, seed=args.seed,
                                   max_videos=args.max_videos)
        print(f"[data] train={len(ds_train)} pairs, val={len(ds_val)} pairs "
              f"(val_frac={args.val_frac})")
    else:
        ds_train = ClevrerChunkPairs(args.cache_dir, seed=args.seed,
                                     max_videos=args.max_videos)
        ds_val = None
        print(f"[data] no split: {len(ds_train)} pairs (overfit mode)")

    pin = (device.type == "cuda")
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=chunk_collate,
                        pin_memory=pin, drop_last=True)
    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=chunk_collate,
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
    ah = AttrsHead(d_static=args.d_static, n_color=N_COLOR, n_material=N_MATERIAL,
                   n_shape=N_SHAPE, hidden=args.attrs_hidden).to(device)
    models = (enc, dec, fwd, eh, ge, gp, ah)
    n_total = sum(sum(p.numel() for p in m.parameters()) for m in models)
    print(f"[model] enc {sum(p.numel() for p in enc.parameters())/1e3:.1f}K | "
          f"dec {sum(p.numel() for p in dec.parameters())/1e6:.2f}M | "
          f"fwd {sum(p.numel() for p in fwd.parameters())} | "
          f"eh {sum(p.numel() for p in eh.parameters())} | "
          f"ge {sum(p.numel() for p in ge.parameters())} | "
          f"gp {sum(p.numel() for p in gp.parameters())} | "
          f"ah {sum(p.numel() for p in ah.parameters())} | "
          f"total {n_total/1e6:.2f}M")

    params = []
    for m in models:
        params += list(m.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.lr_schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"unknown lr_schedule {args.lr_schedule}")

    history = []
    step = 0
    t0 = time.time()
    best_val_recon = float("inf")
    best_epoch = 0
    val_no_improve = 0
    for ep in range(1, args.epochs + 1):
        stage = stage_at_epoch(ep, args.stage1_epochs, args.stage2_epochs)
        sums = {}
        n_batches = 0
        [m.train() for m in models]
        for batch in loader:
            losses = compute_losses(batch, models, args, stage, device)
            total = losses["total"]
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1

            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n_batches += 1
            if args.max_steps > 0 and step >= args.max_steps:
                break
        sched.step()
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}

        val_metrics = None
        if val_loader is not None and (ep % args.val_every == 0 or ep == args.epochs):
            val_metrics = validate(models, val_loader, args, device)

        if ep % args.log_every == 0 or ep == 1:
            keys = ["recon", "pred", "fwd", "consist", "attrs", "event_aux", "gate", "total"]
            train_str = " ".join(f"{k}={avg[k]:.5f}" for k in keys if k in avg)
            diag_str = (f" |z_s_std={avg.get('z_static_std', 0):.3f}"
                        f" |z_d_norm={avg.get('z_dyn_obs_norm', 0):.3f}"
                        f" |z_d_roll={avg.get('z_dyn_roll_norm', 0):.3f}"
                        f" |z_ev={avg.get('z_event_norm', 0):.3f}"
                        f" |gate={avg.get('gate_GT_mean', 0):.3f}")
            val_str = ""
            if val_metrics is not None:
                val_str = (f" | val_recon={val_metrics['recon']:.5f}"
                           f" val_pred={val_metrics['pred']:.5f}"
                           f" val_consist={val_metrics['consist']:.4f}"
                           f" val_z_s_std={val_metrics['z_static_std']:.3f}")
            elapsed = time.time() - t0
            print(f"[ep {ep:3d}/{args.epochs} stage{stage}] {train_str}{diag_str}{val_str} | "
                  f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        history.append({"epoch": ep, "stage": stage, "step": step,
                        "lr": opt.param_groups[0]["lr"],
                        "train": avg, "val": val_metrics})

        # ---- wandb log (always log per-epoch, even when no val ran this epoch) ----
        if use_wandb:
            log_row = {
                "epoch": ep, "stage": stage, "step": step,
                "lr": opt.param_groups[0]["lr"],
                "wall_s": time.time() - t0,
                **{f"train/{k}": v for k, v in avg.items()},
            }
            if val_metrics is not None:
                log_row.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(log_row, step=ep)

        def _build_ckpt():
            return {
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "fwd": fwd.state_dict(),
                "event_head": eh.state_dict(),
                "g_event": ge.state_dict(),
                "gate_predictor": gp.state_dict(),
                "attrs_head": ah.state_dict(),
                "args": vars(args),
                "epoch": ep,
                "step": step,
            }

        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0 or ep == args.epochs):
            torch.save(_build_ckpt(), out_dir / "v5.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # ---- early stopping + best-by-val ckpt ----
        # Only eligible once stage 2 has started — stage 1 ends with low val_recon
        # because InfoNCE/L_pred haven't activated yet; the stage-2 transition
        # bumps val_recon up briefly, and we don't want best_val_recon to lock
        # onto the pre-contrastive ckpt.
        if val_metrics is not None and stage >= 2:
            cur = val_metrics["recon"]
            if cur < best_val_recon - args.early_stop_min_delta:
                best_val_recon = cur
                best_epoch = ep
                val_no_improve = 0
                torch.save(_build_ckpt(), out_dir / "v5_best.pt")
                print(f"  [best] val_recon={cur:.5f} at ep {ep} -> saved v5_best.pt")
                if use_wandb:
                    wandb.log({"val/best_recon": best_val_recon, "val/best_epoch": best_epoch}, step=ep)
            else:
                val_no_improve += 1
                if args.early_stop_patience > 0 and val_no_improve >= args.early_stop_patience:
                    print(f"[stop] early stop: val_recon hasn't improved by "
                          f">={args.early_stop_min_delta} for {val_no_improve} val checks. "
                          f"Best val_recon={best_val_recon:.5f} at ep {best_epoch}.")
                    break

        if args.max_steps > 0 and step >= args.max_steps:
            print(f"[stop] hit max_steps={args.max_steps}")
            break

    # Final ckpt + history
    torch.save(_build_ckpt(), out_dir / "v5.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done] final: {avg}")
    print(f"[best] val_recon={best_val_recon:.5f} at ep {best_epoch} (v5_best.pt)")
    if use_wandb:
        wandb.run.summary["best_val_recon"] = best_val_recon
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()


if __name__ == "__main__":
    main()
