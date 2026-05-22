"""scripts/train_v5_events.py — frozen-encoder events fine-tune (TODO 0.1).

Loads a v5.1.1 checkpoint, freezes the encoder, decoder, ForwardDynamics, and
AttrsHead, and trains only the three event modules (EventHead + GEvent +
GatePredictor) with L_event_aux + L_gate. This is the canonical downstream
probe of z_dyn for the events channel: can collision-event structure be
learned from a frozen z_dyn?

Outputs a checkpoint `v5_events.pt` in the same format as v5.pt — encoder /
decoder / fwd / attrs_head weights are unchanged from the input ckpt; only
event_head / g_event / gate_predictor are updated.

Run:
    python -u scripts/train_v5_events.py \\
        --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \\
        --in_ckpt   outputs/v511_exp1_attrs_20260521_095305/v5_best.pt \\
        --out_dir   outputs/v511_events_<stamp>
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

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.model.attrs_head import AttrsHead
from src.model.event_head import EventHead, GEvent, GatePredictor
from src.model.forward_dynamics import ForwardDynamics
from src.model.latent_decoder import LatentDecoder
from src.model.latent_encoder import LatentEncoder3D


@torch.no_grad()
def compute_event_losses(batch, enc, fwd, eh, ge, gp, device):
    """Frozen-encoder forward pass + event loss assembly.

    Returns (L_event_aux, L_gate, diagnostics) — all event-only modules
    differentiable, encoder/fwd grads disabled.
    """
    chunk_obs   = batch["chunk_obs"].to(device)
    chunk_pred  = batch["chunk_pred"].to(device)
    gate_GT     = batch["gate_GT"].to(device).float()

    # Encoder + fwd are frozen — no grad needed.
    enc_obs  = enc(chunk_obs)
    enc_pred = enc(chunk_pred)
    z_dyn_obs = enc_obs["z_dyn"]
    z_dyn_pred_target = enc_pred["z_dyn"]
    z_dyn_last = z_dyn_obs[:, -1]
    T_chunk = z_dyn_obs.shape[1]
    z_dyn_pred_base = fwd.chunk_step(z_dyn_last, T_chunk)
    return z_dyn_last, z_dyn_pred_base, z_dyn_pred_target, gate_GT, T_chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--in_ckpt",   required=True,
                    help="Existing v5.1.1 ckpt to load (encoder/decoder/fwd weights).")
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--epochs",      type=int, default=10)
    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--lambda_event_aux", type=float, default=1.0)
    ap.add_argument("--lambda_gate",      type=float, default=1.0)
    ap.add_argument("--val_frac",  type=float, default=0.2)
    ap.add_argument("--val_every", type=int,   default=1)
    ap.add_argument("--max_videos", type=int, default=0)
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--wandb_project", default="dialga")
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_mode", default="online",
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(args.wandb_project) and _WANDB_OK and args.wandb_mode != "disabled"
    if use_wandb:
        run_name = args.wandb_run_name or out_dir.name
        os.environ.setdefault("WANDB_DIR", str(out_dir))
        os.environ.setdefault("WANDB_SILENT", "true")
        wandb.init(project=args.wandb_project, name=run_name,
                   mode=args.wandb_mode, config=vars(args), dir=str(out_dir))
        print(f"[wandb] project={args.wandb_project} run={run_name}")

    # ---- data ----
    ds_train = ClevrerChunkPairs(args.cache_dir, split="train",
                                 val_frac=args.val_frac, seed=args.seed,
                                 max_videos=args.max_videos)
    ds_val = ClevrerChunkPairs(args.cache_dir, split="val",
                               val_frac=args.val_frac, seed=args.seed,
                               max_videos=args.max_videos)
    print(f"[data] train={len(ds_train)} val={len(ds_val)} (val_frac={args.val_frac})")
    pin = (device.type == "cuda")
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=chunk_collate,
                        pin_memory=pin, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=chunk_collate,
                            pin_memory=pin)

    # ---- load ckpt and instantiate modules with matching dims ----
    print(f"[ckpt] loading {args.in_ckpt}")
    ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 32))
    d_dyn    = int(a.get("d_dyn", 16))
    d_state  = int(a.get("d_state", 8))
    d_event  = int(a.get("d_event", 4))
    enc_hidden_ch = int(a.get("enc_hidden_ch", 32))
    dec_hidden_ch = int(a.get("dec_hidden_ch", 64))
    chunk_size_lat = int(a.get("chunk_size_lat", 9))
    no_proj = bool(a.get("no_proj", False))
    shared_trunk = bool(a.get("shared_trunk", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn,
                          hidden_ch=enc_hidden_ch,
                          shared_trunk=shared_trunk).to(device)
    dec = LatentDecoder(d_static=d_static, d_dyn=d_dyn,
                        hidden_ch=dec_hidden_ch,
                        chunk_size_lat=chunk_size_lat).to(device)
    fwd = ForwardDynamics(d_dyn=d_dyn, d_state=d_state, no_proj=no_proj).to(device)
    eh = EventHead(d_dyn=d_dyn, d_event=d_event).to(device)
    ge = GEvent(d_event=d_event, d_dyn=d_dyn).to(device)
    gp = GatePredictor(d_dyn=d_dyn).to(device)

    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    fwd.load_state_dict(ckpt["fwd"])
    if "event_head" in ckpt:    eh.load_state_dict(ckpt["event_head"])
    if "g_event" in ckpt:       ge.load_state_dict(ckpt["g_event"])
    if "gate_predictor" in ckpt: gp.load_state_dict(ckpt["gate_predictor"])

    # AttrsHead may be present from Exp 1; preserve weights but freeze.
    ah = None
    if "attrs_head" in ckpt:
        from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
        ah = AttrsHead(d_static=d_static, n_color=len(COLOR_VOCAB),
                       n_material=len(MATERIAL_VOCAB), n_shape=len(SHAPE_VOCAB),
                       hidden=int(a.get("attrs_hidden", 0))).to(device)
        ah.load_state_dict(ckpt["attrs_head"])

    # ---- freeze everything except event modules ----
    for m in [enc, dec, fwd] + ([ah] if ah is not None else []):
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    event_params = list(eh.parameters()) + list(ge.parameters()) + list(gp.parameters())
    opt = torch.optim.AdamW(event_params, lr=args.lr, weight_decay=args.weight_decay)
    n_event = sum(p.numel() for p in event_params)
    print(f"[model] training {n_event} event params | encoder/decoder/fwd frozen")

    # ---- training ----
    history = []
    step = 0
    t0 = time.time()
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        eh.train(); ge.train(); gp.train()
        sums = {"event_aux": 0.0, "gate": 0.0, "total": 0.0, "gate_acc": 0.0}
        n_batches = 0
        for batch in loader:
            z_dyn_last, z_dyn_pred_base, z_dyn_pred_target, gate_GT, T_chunk = (
                compute_event_losses(batch, enc, fwd, eh, ge, gp, device)
            )
            # Event modules are NOT in no_grad — re-run the relevant pieces.
            z_event = eh(z_dyn_last)                                   # (B, D_e)
            pred_residual = ge(z_event)                                # (B, D_d)
            pred_residual_t = pred_residual.unsqueeze(1).expand(-1, T_chunk, -1)
            true_residual = (z_dyn_pred_target - z_dyn_pred_base).detach()
            L_event_aux = (F.mse_loss(pred_residual_t, true_residual, reduction="none")
                           * gate_GT.unsqueeze(-1).unsqueeze(-1)).mean()
            gate_logits = gp(z_dyn_last)
            L_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_GT)
            total = args.lambda_event_aux * L_event_aux + args.lambda_gate * L_gate

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(event_params, 1.0)
            opt.step()
            step += 1

            with torch.no_grad():
                gate_pred_bin = (torch.sigmoid(gate_logits) > 0.5).float()
                gate_acc = (gate_pred_bin == gate_GT).float().mean().item()
            sums["event_aux"] += float(L_event_aux.detach())
            sums["gate"]      += float(L_gate.detach())
            sums["total"]     += float(total.detach())
            sums["gate_acc"]  += gate_acc
            n_batches += 1
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}

        # ---- validation ----
        val_metrics = None
        if ep % args.val_every == 0 or ep == args.epochs:
            eh.eval(); ge.eval(); gp.eval()
            vs = {"event_aux": 0.0, "gate": 0.0, "gate_acc": 0.0, "n": 0}
            tp = fp = fn = tn = 0
            with torch.no_grad():
                for batch in val_loader:
                    z_dyn_last, z_dyn_pred_base, z_dyn_pred_target, gate_GT, T_chunk = (
                        compute_event_losses(batch, enc, fwd, eh, ge, gp, device)
                    )
                    z_event = eh(z_dyn_last); pred_residual = ge(z_event)
                    pred_residual_t = pred_residual.unsqueeze(1).expand(-1, T_chunk, -1)
                    true_residual = z_dyn_pred_target - z_dyn_pred_base
                    L_event_aux = (F.mse_loss(pred_residual_t, true_residual, reduction="none")
                                   * gate_GT.unsqueeze(-1).unsqueeze(-1)).mean()
                    gate_logits = gp(z_dyn_last)
                    L_gate = F.binary_cross_entropy_with_logits(gate_logits, gate_GT)
                    B = gate_GT.shape[0]
                    vs["event_aux"] += float(L_event_aux) * B
                    vs["gate"]      += float(L_gate) * B
                    bin_pred = (torch.sigmoid(gate_logits) > 0.5).float()
                    vs["gate_acc"] += float((bin_pred == gate_GT).float().sum())
                    vs["n"] += B
                    gt_b = gate_GT.bool(); pred_b = bin_pred.bool()
                    tp += int((pred_b & gt_b).sum())
                    fp += int((pred_b & ~gt_b).sum())
                    fn += int((~pred_b & gt_b).sum())
                    tn += int((~pred_b & ~gt_b).sum())
            n = max(vs["n"], 1)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            val_metrics = {"event_aux": vs["event_aux"] / n, "gate": vs["gate"] / n,
                           "gate_acc": vs["gate_acc"] / n,
                           "gate_precision": prec, "gate_recall": rec, "gate_f1": f1}

        if ep % args.log_every == 0 or ep == 1:
            elapsed = time.time() - t0
            val_str = ""
            if val_metrics is not None:
                val_str = (f" | val_event={val_metrics['event_aux']:.5f}"
                           f" val_gate={val_metrics['gate']:.4f}"
                           f" val_gate_acc={val_metrics['gate_acc']:.3f}"
                           f" val_F1={val_metrics['gate_f1']:.3f}")
            print(f"[ep {ep:3d}/{args.epochs}] "
                  f"event_aux={avg['event_aux']:.5f} gate={avg['gate']:.4f} "
                  f"total={avg['total']:.4f} gate_acc={avg['gate_acc']:.3f}"
                  f"{val_str} | lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        history.append({"epoch": ep, "step": step, "train": avg, "val": val_metrics})

        if use_wandb:
            row = {"epoch": ep, "step": step, "wall_s": time.time() - t0,
                   **{f"train/{k}": v for k, v in avg.items()}}
            if val_metrics is not None:
                row.update({f"val/{k}": v for k, v in val_metrics.items()})
            wandb.log(row, step=ep)

        def _build_ckpt():
            d = {"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                 "fwd": fwd.state_dict(), "event_head": eh.state_dict(),
                 "g_event": ge.state_dict(), "gate_predictor": gp.state_dict(),
                 "args": {**vars(args), **{k: a[k] for k in a if k not in vars(args)}},
                 "epoch": ep, "step": step}
            if ah is not None:
                d["attrs_head"] = ah.state_dict()
            return d

        torch.save(_build_ckpt(), out_dir / "v5_events.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val_metrics is not None:
            cur = val_metrics["event_aux"]
            if cur < best_val:
                best_val = cur
                torch.save(_build_ckpt(), out_dir / "v5_events_best.pt")

    print(f"\n[done] best val_event_aux={best_val:.5f}; ckpt → {out_dir/'v5_events_best.pt'}")


if __name__ == "__main__":
    main()
