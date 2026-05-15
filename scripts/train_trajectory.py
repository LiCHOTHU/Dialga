"""Train the 4-component trajectory latent on cached Wan-VAE latents.

Loads windows from a cache produced by `cache_wan_latents.py`, trains
TrajectoryEncoder + DynPredictor + TrajectoryDecoder with PV-VAE-style
random frame masking and the single trajectory_loss family.

The frozen Wan-VAE pixel decoder is NOT used in this script — we operate
entirely in latent space. Pixel rendering happens at eval time with a
separate script (decode cached latents through the Wan VAE).

Usage:
    python scripts/train_trajectory.py \\
        --cache_dir outputs/cache/wan_5vid_W12 \\
        --out_dir outputs/traj5_5vid \\
        --epochs 200 --batch_size 4 --mask_ratio 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.trajectory_encoder import (
    TrajectoryEncoder, TrajectoryDecoder, DynPredictor, fill_z_dyn,
    event_to_alpha, trajectory_loss, slot_contrast_loss,
)


class CachedLatentDataset(Dataset):
    """Loads pre-encoded Wan latents from `cache_wan_latents.py` output.

    Supports train/val split BY VIDEO_ID (not by window). This is important —
    a per-window split would leak between train and val because windows from
    the same video share scene-level statistics. Splitting by video_id
    means train and val are entirely disjoint at the video level.
    """

    def __init__(self, cache_dir, split: str = "all", val_frac: float = 0.0,
                 seed: int = 42):
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "metadata.json").read_text())
        windows = meta["windows"]
        if val_frac > 0 and split in ("train", "val"):
            vids = sorted({w["video_id"] for w in windows})
            import random as _r
            rng = _r.Random(seed)
            rng.shuffle(vids)
            n_val = max(1, int(len(vids) * val_frac))
            val_vids = set(vids[:n_val])
            if split == "train":
                windows = [w for w in windows if w["video_id"] not in val_vids]
            else:  # val
                windows = [w for w in windows if w["video_id"] in val_vids]
        self.cache_dir = cache_dir
        self.windows = windows
        self.window_length = int(meta["args"]["window_length"])
        self.split = split

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path = self.cache_dir / self.windows[idx]["path"]
        blob = torch.load(path, map_location="cpu", weights_only=False)
        # Optionally load co-located DINOv2 features (DINOSAUR auxiliary).
        # File layout: <cache_dir>/dino/<window_idx>.pt with key "cls".
        # The window_idx is the same numeric index as in the cache filename.
        w = self.windows[idx]
        win_idx = int(Path(w["path"]).stem)
        dino_path = self.cache_dir / "dino" / f"{win_idx:06d}.pt"
        if dino_path.exists():
            d = torch.load(dino_path, map_location="cpu", weights_only=False)
            blob["dino_cls"] = d["cls"]                       # (T_pix, 384)
        return blob


def collate(batch):
    """Stack the latent tensors and per-window metadata."""
    out = {}
    out["latent"] = torch.stack([b["latent"] for b in batch])
    out["positions"] = torch.stack([b["positions"] for b in batch])
    out["visibility"] = torch.stack([b["visibility"] for b in batch])
    out["slot_mask"] = torch.stack([b["slot_mask"] for b in batch])
    out["attrs"] = torch.stack([b["attrs"] for b in batch])
    out["collision_mask"] = torch.stack([b["collision_mask"] for b in batch])
    out["video_id"] = torch.tensor([b["video_id"] for b in batch], dtype=torch.long)
    out["collisions"] = [b["collisions"] for b in batch]
    if "dino_cls" in batch[0]:
        out["dino_cls"] = torch.stack([b["dino_cls"] for b in batch])    # (B, T_pix, 384)
    return out


def sample_frame_mask(B: int, T: int, ratio: float, always_visible: tuple = (0,),
                      device: torch.device | None = None) -> torch.Tensor:
    """Random per-frame visibility mask. True = visible to encoder.

    `ratio` is the probability of hiding each frame. `always_visible` lists
    frame indices that must always be visible (default: frame 0 — gives the
    scene/static branches a guaranteed anchor frame, à la PV-VAE).
    """
    vis = torch.rand(B, T, device=device) > ratio
    for t in always_visible:
        if 0 <= t < T:
            vis[:, t] = True
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout in transformer layers (encoder + decoder).")
    ap.add_argument("--mask_ratio", type=float, default=0.4,
                    help="Per-frame mask probability (PV-VAE style). 0 = no masking.")

    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_static", type=int, default=16)
    ap.add_argument("--d_dyn", type=int, default=32)

    ap.add_argument("--lambda_smooth", type=float, default=0.1)
    ap.add_argument("--lambda_entropy", type=float, default=0.01)
    ap.add_argument("--lambda_vicreg", type=float, default=0.01)
    ap.add_argument("--lambda_diff", type=float, default=0.0,
                    help="PV-VAE motion-aware MSE on temporal differences. "
                         "Heavily penalizes 'render-the-mean' collapse.")
    ap.add_argument("--lambda_alpha_sparse", type=float, default=0.0,
                    help="L1 on cumulative α. Pushes α=0 at frames where slot "
                         "isn't needed → unsupervised mid-clip-entry localization.")
    ap.add_argument("--lambda_slot_bw", type=float, default=0.0,
                    help="Rate-distortion: mean(α · ‖z_dyn‖²). Penalizes a slot "
                         "only when it's alive AND has non-trivial content. "
                         "Principled lever for unsupervised event localization.")
    ap.add_argument("--use_null", action="store_true",
                    help="Enable the 'never born' null option in the event softmax. "
                         "Default off — appropriate for overfit / always-present-objects.")
    ap.add_argument("--hard_alpha", action="store_true",
                    help="Use straight-through Gumbel-softmax → α ∈ {0, 1}. "
                         "Forces discrete event decisions; removes the soft-α "
                         "escape hatch that lets the model avoid localization.")
    ap.add_argument("--gumbel_tau", type=float, default=0.5,
                    help="Gumbel temperature when --hard_alpha is set.")
    ap.add_argument("--lambda_event_sup", type=float, default=0.0,
                    help="GT-visibility supervision on event channel. NLL loss "
                         "between event_prob and GT visibility-onset frame. "
                         "Only mechanism that reliably gives unsupervised-fail "
                         "event-localization (after 5 failed unsupervised tries).")
    ap.add_argument("--val_frac", type=float, default=0.0,
                    help="Fraction of videos held out for validation (by video_id). "
                         "0 = no split (overfit mode); 0.2 = 80/20 split.")
    ap.add_argument("--val_every", type=int, default=10,
                    help="Run validation every N epochs.")
    ap.add_argument("--lambda_dino", type=float, default=0.0,
                    help="DINOSAUR-style auxiliary loss: MSE between projected "
                         "z_static and visibility-weighted DINOv2 CLS features. "
                         "Forces z_static to be semantically linearly-readable.")
    ap.add_argument("--d_dino", type=int, default=384,
                    help="DINOv2 feature dim (DINOv2-S = 384).")
    ap.add_argument("--lambda_contrast", type=float, default=0.0,
                    help="CoSA slot-contrastive weight (0 = off). Requires --mask_ratio > 0.")
    ap.add_argument("--contrast_tau", type=float, default=0.1)

    ap.add_argument("--use_predictor", action="store_true",
                    help="Fill z_dyn at masked positions via DynPredictor before decoding.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=50)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.val_frac > 0:
        ds_train = CachedLatentDataset(args.cache_dir, split="train", val_frac=args.val_frac)
        ds_val = CachedLatentDataset(args.cache_dir, split="val", val_frac=args.val_frac)
        print(f"[data] split by video_id: train={len(ds_train)} windows, "
              f"val={len(ds_val)} windows  (val_frac={args.val_frac})")
        ds = ds_train  # the loader iterates the train set
    else:
        ds = CachedLatentDataset(args.cache_dir)
        ds_val = None
        print(f"[data] no split: {len(ds)} cached windows (overfit mode)")
    print(f"[data] window length: {ds.window_length} frames")
    sample = ds[0]
    C, T_lat, H_lat, W_lat = sample["latent"].shape
    print(f"[data] latent shape per window: ({C}, {T_lat}, {H_lat}, {W_lat})")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=(device.type == "cuda"))
    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate,
                                pin_memory=(device.type == "cuda"))

    enc = TrajectoryEncoder(
        latent_ch=C, K=args.K, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, T_max=max(T_lat * 2, 16),
        spatial_size=H_lat,
        d_static=args.d_static, d_dyn=args.d_dyn,
        dropout=args.dropout,
    ).to(device)
    dec = TrajectoryDecoder(
        latent_ch=C, K=args.K, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, T_max=max(T_lat * 2, 16),
        spatial_size=H_lat,
        d_static=args.d_static, d_dyn=args.d_dyn,
        dropout=args.dropout,
    ).to(device)
    pred = None
    if args.use_predictor:
        pred = DynPredictor(d_dyn=args.d_dyn, d_static=args.d_static).to(device)

    # DINOSAUR-style alignment head: project z_static to DINOv2 feature dim
    # and align via MSE with per-slot visibility-weighted DINO CLS targets.
    dino_head = None
    if args.lambda_dino > 0:
        dino_head = nn.Sequential(
            nn.Linear(args.d_static, args.d_static * 2),
            nn.SiLU(),
            nn.Linear(args.d_static * 2, args.d_dino),
        ).to(device)
        print(f"[dino] alignment head ON (d_static={args.d_static} → {args.d_dino}, "
              f"λ_dino={args.lambda_dino})")

    n_enc = sum(p.numel() for p in enc.parameters())
    n_dec = sum(p.numel() for p in dec.parameters())
    n_pred = sum(p.numel() for p in pred.parameters()) if pred is not None else 0
    print(f"[model] enc {n_enc/1e6:.2f}M | dec {n_dec/1e6:.2f}M | pred {n_pred/1e3:.1f}K "
          f"| total {(n_enc + n_dec + n_pred)/1e6:.2f}M")

    params = list(enc.parameters()) + list(dec.parameters())
    if pred is not None:
        params = params + list(pred.parameters())
    if dino_head is not None:
        params = params + list(dino_head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    history = {"epoch": [], "recon": [], "smooth": [], "entropy": [], "var": [], "cov": []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        enc.train(); dec.train()
        if pred is not None:
            pred.train()
        sums = {"recon": 0.0, "diff": 0.0, "smooth": 0.0, "entropy": 0.0, "var": 0.0, "cov": 0.0,
                "total": 0.0, "mean_null": 0.0, "mean_alpha_last": 0.0}
        n_batches = 0
        for batch in loader:
            target = batch["latent"].to(device)                          # (B, C, T, H, W)
            B = target.shape[0]; T = target.shape[2]
            if args.mask_ratio > 0:
                mask = sample_frame_mask(B, T, args.mask_ratio, always_visible=(0,), device=device)
            else:
                mask = torch.ones(B, T, dtype=torch.bool, device=device)

            z_static, z_dyn, event_logits, null_logits = enc(target, frame_mask=mask)
            event_prob, null_prob, alpha = event_to_alpha(
                event_logits, null_logits,
                use_null=args.use_null,
                hard=args.hard_alpha,
                tau=args.gumbel_tau,
            )

            if pred is not None and args.use_predictor:
                z_dyn_for_decoder = fill_z_dyn(z_dyn, pred, z_static, mask)
            else:
                z_dyn_for_decoder = z_dyn

            recon = dec(z_static, z_dyn_for_decoder, alpha)

            losses = trajectory_loss(
                z_static, z_dyn_for_decoder, event_prob, null_prob, alpha, recon, target,
                lambda_smooth=args.lambda_smooth,
                lambda_entropy=args.lambda_entropy,
                lambda_vicreg=args.lambda_vicreg,
                lambda_diff=args.lambda_diff,
                lambda_alpha_sparse=args.lambda_alpha_sparse,
                lambda_slot_bw=args.lambda_slot_bw,
            )
            loss = losses["total"]

            # DINOSAUR auxiliary: align projected z_static with visibility-weighted DINOv2 CLS
            dino_loss_val = 0.0
            if args.lambda_dino > 0 and dino_head is not None and "dino_cls" in batch:
                dino_cls = batch["dino_cls"].to(device)               # (B, T_pix, 384)
                visib = batch["visibility"].to(device).float()        # (B, T_pix, K)
                slot_mask_b = batch["slot_mask"].to(device).float()    # (B, K)
                w_sum = visib.sum(dim=1).clamp_min(1e-3)               # (B, K)
                # Per-slot target = visibility-weighted mean DINO CLS over time
                target_dino = torch.einsum("btk,btd->bkd", visib, dino_cls) / w_sum.unsqueeze(-1)
                pred_dino = dino_head(z_static)                        # (B, K, d_dino)
                err = (pred_dino - target_dino).pow(2).mean(dim=-1)    # (B, K)
                if slot_mask_b.sum() > 0:
                    dino_loss = (err * slot_mask_b).sum() / slot_mask_b.sum()
                    loss = loss + args.lambda_dino * dino_loss
                    dino_loss_val = float(dino_loss.detach())

            # GT-visibility event supervision (weak supervision)
            event_sup_loss_val = 0.0
            if args.lambda_event_sup > 0:
                visib = batch["visibility"].to(device).bool()        # (B, T_pix, K)
                slot_mask_b = batch["slot_mask"].to(device).bool()    # (B, K)
                B_, T_pix, K_ = visib.shape
                any_visible = visib.any(dim=1)                        # (B, K)
                # first-visible pixel frame per (B, K)
                # argmax of bool gives first True position (or 0 if no True)
                first_pix = visib.int().argmax(dim=1)                 # (B, K)
                # Map to latent frame index
                first_lat = (first_pix * T // T_pix).clamp(0, T - 1)  # (B, K)
                # event_prob shape (B, T, K); we want -log p[first_lat] per (b,k)
                log_ev = (event_prob.clamp_min(1e-8)).log()           # (B, T, K)
                nll = -log_ev.gather(1, first_lat.unsqueeze(1).long()).squeeze(1)  # (B, K)
                # Mask: only real visible slots
                m = (slot_mask_b & any_visible).float()
                if m.sum() > 0:
                    event_sup_loss = (nll * m).sum() / m.sum()
                    loss = loss + args.lambda_event_sup * event_sup_loss
                    event_sup_loss_val = float(event_sup_loss.detach())

            # ---- additive literature-driven losses ----
            extras = {}
            if args.lambda_contrast > 0 and args.mask_ratio > 0:
                # Second encoder pass with a DIFFERENT random mask for view2
                mask2 = sample_frame_mask(B, T, args.mask_ratio, always_visible=(0,), device=device)
                z_static_v2, _, _, _ = enc(target, frame_mask=mask2)
                contrast = slot_contrast_loss(z_static, z_static_v2, tau=args.contrast_tau)
                loss = loss + args.lambda_contrast * contrast
                extras["contrast"] = float(contrast.detach())

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

            for k in sums:
                if k in losses:
                    sums[k] += float(losses[k].item()) if torch.is_tensor(losses[k]) else float(losses[k])
            sums["total"] += float(loss.item())
            n_batches += 1
        sched.step()
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}

        history["epoch"].append(ep)
        for k in ("recon", "smooth", "entropy", "var", "cov"):
            history[k].append(avg[k])

        # Validation pass (no grad, mean MSE on val set)
        val_recon = None
        if val_loader is not None and (ep % args.val_every == 0 or ep == args.epochs):
            enc.eval(); dec.eval()
            with torch.no_grad():
                v_sum = 0.0; v_n = 0
                for batch in val_loader:
                    target = batch["latent"].to(device)
                    Bv, Tv = target.shape[0], target.shape[2]
                    mask = torch.ones(Bv, Tv, dtype=torch.bool, device=device)
                    z_static, z_dyn, event_logits, null_logits = enc(target, frame_mask=mask)
                    event_prob, null_prob, alpha = event_to_alpha(
                        event_logits, null_logits, use_null=args.use_null,
                        hard=False, tau=args.gumbel_tau)
                    if pred is not None and args.use_predictor:
                        z_dyn_for_decoder = fill_z_dyn(z_dyn, pred, z_static, mask)
                    else:
                        z_dyn_for_decoder = z_dyn
                    recon_v = dec(z_static, z_dyn_for_decoder, alpha)
                    v_sum += F.mse_loss(recon_v, target, reduction="mean").item() * Bv
                    v_n += Bv
                val_recon = v_sum / max(v_n, 1)
            enc.train(); dec.train()

        if ep % args.log_every == 0 or ep == 1 or val_recon is not None:
            elapsed = time.time() - t0
            val_str = f" | val_recon {val_recon:.5f}" if val_recon is not None else ""
            print(f"[ep {ep:4d}/{args.epochs}] "
                  f"total {avg['total']:.5f} | recon {avg['recon']:.5f}{val_str} | "
                  f"smooth {avg['smooth']:.5f} | entropy {avg['entropy']:.4f} | "
                  f"var {avg['var']:.5f} | cov {avg['cov']:.5f} | "
                  f"lr {optim.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0 or ep == args.epochs):
            ckpt = {
                "encoder_state_dict": enc.state_dict(),
                "decoder_state_dict": dec.state_dict(),
                "args": vars(args),
                "epoch": ep,
                "history": history,
            }
            if pred is not None:
                ckpt["predictor_state_dict"] = pred.state_dict()
            ckpt_path = out_dir / "trajectory.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  saved → {ckpt_path}")

    print(f"\n[done] final recon = {history['recon'][-1]:.6f}")


if __name__ == "__main__":
    main()
