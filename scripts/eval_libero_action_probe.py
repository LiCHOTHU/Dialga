"""Action-prediction probe for the LIBERO-trained v5.1.2 encoder.

The promise of z_dyn is that it carries USEFUL motion structure — not just the
collision toy on CLEVRER but the actual robot action signal on LIBERO. This
probe trains a tiny MLP on FROZEN z_dyn to predict per-frame actions, then
compares it against fair baselines.

  Inputs : z_dyn  (B, T_lat=9, D_d) — encoder output for chunk_obs
  Target : actions averaged into T_lat windows of 4 frames -> (B, 9, 7)
  Loss   : MSE on the continuous 6 dims + BCE on the binary gripper (dim 6).

The MLP is shared across timesteps (Linear(D_d, hidden) -> ReLU -> Linear(hidden, 7))
so we measure the LATENT's informativeness, not the head's capacity.

Splits ----------------------------------------------------------------------
  - In-task generalization : non-holdout tasks, evaluated on val episodes
    (episode_id %% val_every_k_episodes == 0).
  - Cross-task generalization : the n_holdout_tasks tasks held out from
    representation training; the probe MLP itself trains on train tasks only.

Baselines (same MLP, different feature) ------------------------------------
  - z_dyn               — the claim
  - z_static (broadcast)— wrong tool: identity, not motion (should fail)
  - wan_mean            — raw Wan latent (C=48) spatially-meaned per frame
  - random_init_z_dyn   — same encoder arch but never trained (capacity sanity)

Metrics --------------------------------------------------------------------
  - cont_mse       : mean MSE on dims 0..5
  - gripper_acc    : binary classification accuracy on sign(action[..., 6])
  - cont_r2_per_d  : per-dim R² on continuous dims (paper-grade table)
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.libero_window import LiberoChunkPairs, libero_collate
from src.model.latent_encoder import LatentEncoder3D


T_LAT = 9
T_PIX = 33
ACTION_DIM = 7
CONT_DIMS = list(range(6))  # 0..5 = xyz + rpy
GRIPPER_DIM = 6


def pool_actions_to_lat(actions: torch.Tensor) -> torch.Tensor:
    """(B, 33, 7) per-pixel-frame -> (B, 9, 7) per-latent-frame, mean-pooled.

    Wan VAE's temporal stride: T_lat = (T_pix - 1)//4 + 1 = 9. Latent frame 0
    corresponds to pixel frame 0; latent frames 1..8 each span 4 pixel frames.
    """
    B, T, A = actions.shape
    out = torch.zeros(B, T_LAT, A, device=actions.device, dtype=actions.dtype)
    out[:, 0] = actions[:, 0]
    for t in range(1, T_LAT):
        lo, hi = 1 + 4 * (t - 1), 1 + 4 * t
        out[:, t] = actions[:, lo:hi].mean(dim=1)
    return out


# ---------- feature extractors ----------------------------------------------

class FeatureExtractor:
    """Picks one of {z_dyn, z_static, wan_mean, random_init_z_dyn} from a batch.

    Returns (B, T_lat, D_feat). z_static is broadcast across T_lat.
    """
    def __init__(self, kind: str, enc: nn.Module | None = None,
                 enc_random: nn.Module | None = None):
        self.kind = kind
        self.enc = enc
        self.enc_random = enc_random

    @torch.no_grad()
    def __call__(self, batch, device) -> torch.Tensor:
        chunk_obs = batch["chunk_obs"].to(device)              # (B, 48, 9, 8, 8)
        B = chunk_obs.shape[0]
        if self.kind == "z_dyn":
            return self.enc(chunk_obs)["z_dyn"].float()        # (B, 9, D_d)
        if self.kind == "z_static":
            zs = self.enc(chunk_obs)["z_static"].float()       # (B, D_s)
            return zs.unsqueeze(1).expand(-1, T_LAT, -1).contiguous()
        if self.kind == "wan_mean":
            return chunk_obs.float().mean(dim=(3, 4)).permute(0, 2, 1).contiguous()
        if self.kind == "wan_flat":
            # (B, 48, 9, 8, 8) -> (B, 9, 48*8*8=3072): keep all spatial info
            x = chunk_obs.float().permute(0, 2, 1, 3, 4)        # (B, 9, 48, 8, 8)
            return x.reshape(B, T_LAT, -1).contiguous()
        if self.kind == "random_init_z_dyn":
            return self.enc_random(chunk_obs)["z_dyn"].float()
        raise ValueError(self.kind)


# ---------- probe head ------------------------------------------------------

class ActionMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------- losses + metrics ------------------------------------------------

def action_loss(pred: torch.Tensor, target: torch.Tensor) -> dict:
    cont_mse = F.mse_loss(pred[..., CONT_DIMS], target[..., CONT_DIMS])
    grip_target = (target[..., GRIPPER_DIM] > 0).float()
    grip_bce = F.binary_cross_entropy_with_logits(pred[..., GRIPPER_DIM], grip_target)
    return {"cont_mse": cont_mse, "grip_bce": grip_bce,
            "total": cont_mse + grip_bce}


@torch.no_grad()
def eval_predictions(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    cont_p = preds[..., CONT_DIMS]
    cont_t = targets[..., CONT_DIMS]
    cont_mse = ((cont_p - cont_t) ** 2).mean().item()
    var = cont_t.var(dim=(0, 1))
    mse_per_d = ((cont_p - cont_t) ** 2).mean(dim=(0, 1))
    r2 = (1.0 - mse_per_d / var.clamp_min(1e-8)).tolist()

    grip_logit = preds[..., GRIPPER_DIM]
    grip_target = (targets[..., GRIPPER_DIM] > 0).float()
    grip_pred = (grip_logit > 0).float()
    grip_acc = (grip_pred == grip_target).float().mean().item()
    grip_pos_rate = grip_target.mean().item()
    return {"cont_mse": cont_mse, "cont_r2_per_d": r2,
            "gripper_acc": grip_acc, "gripper_pos_rate": grip_pos_rate}


# ---------- main ------------------------------------------------------------

def build_encoder_from_ckpt(ckpt: dict, device) -> LatentEncoder3D:
    a = ckpt["args"]
    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc = LatentEncoder3D(
        d_static=int(a.get("d_static", 64)),
        d_dyn=int(a.get("d_dyn", 64)),
        hidden_ch=int(a.get("enc_hidden_ch", 128)),
        shared_trunk=bool(a.get("shared_trunk", False)),
        use_layer_norm=use_ln,
    ).to(device).eval()
    enc.load_state_dict(ckpt["encoder"])
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def build_random_encoder_like(enc: LatentEncoder3D, device) -> LatentEncoder3D:
    rnd = LatentEncoder3D(
        d_static=enc.d_static, d_dyn=enc.d_dyn,
        hidden_ch=enc.hidden_ch,
        shared_trunk=enc.shared_trunk,
        use_layer_norm=enc.use_layer_norm,
    ).to(device).eval()
    for p in rnd.parameters():
        p.requires_grad_(False)
    return rnd


def train_one_probe(kind: str, fx: FeatureExtractor, train_loader, val_loader,
                    holdout_loader, d_in: int, args, device) -> dict:
    mlp = ActionMLP(d_in, hidden=args.probe_hidden).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=args.probe_lr,
                            weight_decay=1e-4)

    t0 = time.time()
    for ep in range(1, args.probe_epochs + 1):
        mlp.train()
        for batch in train_loader:
            feat = fx(batch, device)                            # (B, 9, D_in)
            act_lat = pool_actions_to_lat(batch["actions_obs"].to(device))
            pred = mlp(feat)
            losses = action_loss(pred, act_lat)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            opt.step()

    def _eval(loader):
        mlp.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                feat = fx(batch, device)
                act_lat = pool_actions_to_lat(batch["actions_obs"].to(device))
                preds.append(mlp(feat).cpu())
                targets.append(act_lat.cpu())
        if not preds: return None
        return eval_predictions(torch.cat(preds), torch.cat(targets))

    return {
        "feature": kind,
        "d_in": d_in,
        "train": _eval(train_loader),
        "val": _eval(val_loader),
        "heldout": _eval(holdout_loader),
        "wall_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True,
                    help="LIBERO Wan-latent cache (encode_libero_wan.py output)")
    ap.add_argument("--ckpt", required=True,
                    help="train_v5_libero.py checkpoint (.pt) — encoder state.")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default <ckpt_dir>/eval_action_probe.json")
    ap.add_argument("--n_holdout_tasks", type=int, default=10,
                    help="Must match what training used.")
    ap.add_argument("--task_holdout_seed", type=int, default=0)
    ap.add_argument("--val_every_k_episodes", type=int, default=10)
    ap.add_argument("--require_chunks_per_episode", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--probe_epochs", type=int, default=20)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", nargs="+",
                    default=["z_dyn", "z_static", "wan_mean", "random_init_z_dyn"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    enc = build_encoder_from_ckpt(ckpt, device)
    enc_random = build_random_encoder_like(enc, device)
    print(f"[ckpt] {args.ckpt}")
    print(f"[enc ] d_static={enc.d_static} d_dyn={enc.d_dyn} "
          f"hidden_ch={enc.hidden_ch} use_ln={enc.use_layer_norm}")

    ds_kw = dict(
        n_holdout_tasks=args.n_holdout_tasks,
        task_holdout_seed=args.task_holdout_seed,
        val_every_k=args.val_every_k_episodes,
        pair_seed=args.seed,
        require_chunks_per_video=args.require_chunks_per_episode,
    )
    ds_train   = LiberoChunkPairs(args.cache_dir, split="train",   **ds_kw)
    ds_val     = LiberoChunkPairs(args.cache_dir, split="val",     **ds_kw)
    ds_holdout = LiberoChunkPairs(args.cache_dir, split="heldout", **ds_kw)
    print(f"[data] train={len(ds_train)} pairs / "
          f"val={len(ds_val)} pairs / heldout={len(ds_holdout)} pairs "
          f"(heldout tasks={sorted(ds_train.holdout_task_ids)})")

    pin = device.type == "cuda"
    def loader_of(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, collate_fn=libero_collate,
                          pin_memory=pin, drop_last=shuffle)
    train_loader = loader_of(ds_train, shuffle=True)
    val_loader   = loader_of(ds_val,     shuffle=False)
    heldout_loader = loader_of(ds_holdout, shuffle=False) if len(ds_holdout) > 0 else None

    feat_dims = {
        "z_dyn": enc.d_dyn,
        "z_static": enc.d_static,
        "wan_mean": 48,
        "wan_flat": 48 * 8 * 8,
        "random_init_z_dyn": enc.d_dyn,
    }
    results = {}
    for kind in args.features:
        d_in = feat_dims[kind]
        fx = FeatureExtractor(kind, enc=enc, enc_random=enc_random)
        print(f"\n=== probe: {kind}  (d_in={d_in}) ===")
        r = train_one_probe(kind, fx, train_loader, val_loader, heldout_loader,
                            d_in, args, device)
        results[kind] = r
        for split in ("train", "val", "heldout"):
            m = r[split]
            if m is None:
                print(f"  {split:8s}  (no data)")
                continue
            r2_str = "[" + ", ".join(f"{x:+.2f}" for x in m["cont_r2_per_d"]) + "]"
            print(f"  {split:8s}  cont_mse={m['cont_mse']:.5f}  "
                  f"gripper_acc={m['gripper_acc']:.3f}  R²={r2_str}")

    # ---------- summary table ----------
    print("\n=== Summary: cont_mse / gripper_acc by split ===")
    print(f"{'feature':<20}{'train_mse':>10}  {'val_mse':>10}  {'hold_mse':>10}  "
          f"{'train_grip':>11}  {'val_grip':>10}  {'hold_grip':>10}")
    for kind in args.features:
        r = results[kind]
        def _cell(d, k):
            return f"{d[k]:>.5f}" if d is not None else "    -"
        def _gcell(d):
            return f"{d['gripper_acc']:>.3f}" if d is not None else "  -"
        print(f"{kind:<20}"
              f"{_cell(r['train'], 'cont_mse'):>10}  "
              f"{_cell(r['val'], 'cont_mse'):>10}  "
              f"{_cell(r['heldout'], 'cont_mse'):>10}  "
              f"{_gcell(r['train']):>11}  "
              f"{_gcell(r['val']):>10}  "
              f"{_gcell(r['heldout']):>10}")

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "eval_action_probe.json"
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "cache_dir": args.cache_dir,
        "args": vars(args),
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_heldout": len(ds_holdout),
        "heldout_task_ids": sorted(ds_train.holdout_task_ids),
        "results": results,
    }, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
