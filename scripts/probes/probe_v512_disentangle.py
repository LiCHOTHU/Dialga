"""probe_v512_disentangle.py — what do z_static / z_dyn / z_event encode?

One pass over the model's held-out val split (the SAME 1000-video split it
trained on: max_videos=1000, val_frac=0.2, seed=42) produces three latents per
chunk. We then fit tiny linear probes and read off a factorization matrix:

                 identity (attr acc)     motion (pos/vel R²)    collision (acc)
   z_static      HIGH  (it's identity)   LOW   (disentangled)   LOW
   z_dyn_last    LOW   (disentangled)    HIGH  (it's motion)    some (motion→hit)
   z_event       --                      --                     untrained → chance

The OFF-DIAGONAL is the disentanglement evidence: z_static should predict
identity but NOT motion; z_dyn should predict motion but NOT identity.

z_event is computed from EventHead(z_dyn_last). In the v5.1.2 production config
EventHead receives no gradient (stage-3 events were removed), so its collision
probe is expected near chance — included to confirm the channel is inert.

Probes: linear, 50/50 split BY video_id (seed 0), disjoint train/test.
  identity   -> classification acc vs majority baseline   (per video)
  motion     -> R² of linear regression                   (per video×chunk)
  collision  -> classification acc vs majority baseline   (per video×chunk)

NOTE (e2 / encoder-unfrozen): this probes the small encoder on CACHED latents
(the original Wan encoder's output). e2 trained its small encoder on latents
re-encoded through the *trained* Wan encoder, so e2's numbers here are a lower
bound — a faithful e2 probe would re-encode pixels first. e1/e3 consume cached
latents natively, so they are exact.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from src.data.clevrer_states import COLOR_VOCAB, MATERIAL_VOCAB, SHAPE_VOCAB
from src.model.latent_encoder import LatentEncoder3D
from src.model.event_head import EventHead


CHUNK_STARTS = [0, 33, 66]
ATTR_GROUPS = [
    ("color",    0,                                      len(COLOR_VOCAB)),
    ("material", len(COLOR_VOCAB),                       len(COLOR_VOCAB) + len(MATERIAL_VOCAB)),
    ("shape",    len(COLOR_VOCAB) + len(MATERIAL_VOCAB), len(COLOR_VOCAB) + len(MATERIAL_VOCAB) + len(SHAPE_VOCAB)),
]


def _chunk_idx_for_start(s: int) -> int:
    return CHUNK_STARTS.index(s)


def modal_label(attrs: torch.Tensor, slot_mask: torch.Tensor, lo: int, hi: int) -> int:
    real_k = slot_mask.bool()
    if not real_k.any():
        return -1
    classes = attrs[real_k, lo:hi].argmax(dim=-1)
    return int(torch.bincount(classes, minlength=hi - lo).argmax().item())


def _scene_state(gt_entry, chunk_idx: int):
    pos = gt_entry["per_chunk_positions"][chunk_idx]                 # (K, 2)
    vel = gt_entry["per_chunk_velocities"][chunk_idx]
    vis = gt_entry["per_chunk_visibility"][chunk_idx].float().unsqueeze(-1)
    denom = vis.sum().clamp_min(1.0)
    return (pos * vis).sum(0) / denom, (vel * vis).sum(0) / denom


@torch.no_grad()
def collect(enc, eh, loader, gt, device):
    """Per (video, chunk): z_static, z_dyn_last, z_event, pos, vel, collision,
    plus per-video attrs/slot_mask."""
    per_chunk = {}
    per_video = {}
    for batch in loader:
        chunk_obs = batch["chunk_obs"].to(device)
        out = enc(chunk_obs)
        z_static = out["z_static"].float().cpu()
        z_dyn_last = out["z_dyn"][:, -1].float().cpu()
        z_event = eh(out["z_dyn"][:, -1]).float().cpu() if eh is not None else None
        for b in range(chunk_obs.shape[0]):
            vid = int(batch["video_id"][b]); s = int(batch["start_frame"][b])
            if vid not in gt:
                continue
            try:
                ci = _chunk_idx_for_start(s)
            except ValueError:
                continue
            per_video.setdefault(vid, {
                "attrs": batch["attrs"][b].clone(),
                "slot_mask": batch["slot_mask"][b].clone().bool(),
                "zs": [], "zd": [],
            })
            per_video[vid]["zs"].append(z_static[b])
            per_video[vid]["zd"].append(z_dyn_last[b])
            key = (vid, ci)
            if key in per_chunk:
                continue
            pos, vel = _scene_state(gt[vid], ci)
            per_chunk[key] = {
                "z_static": z_static[b], "z_dyn_last": z_dyn_last[b],
                "z_event": (z_event[b] if z_event is not None else None),
                "pos": pos, "vel": vel,
                "collision": int(gt[vid]["chunk_collision"][ci].item()),
            }
    # finalize per-video averaged features
    for vid, d in per_video.items():
        d["z_static"] = torch.stack(d["zs"]).mean(0)
        d["z_dyn_last"] = torch.stack(d["zd"]).mean(0)
        del d["zs"], d["zd"]
    return per_chunk, per_video


def lin_classify(x_tr, y_tr, x_te, y_te, n_cls, epochs, lr, wd, device):
    probe = nn.Linear(x_tr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    x_tr, y_tr, x_te, y_te = (t.to(device) for t in (x_tr, y_tr, x_te, y_te))
    best = {"train_acc": 0.0, "test_acc": 0.0, "loss": float("inf")}
    for ep in range(epochs):
        probe.train()
        loss = F.cross_entropy(probe(x_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            probe.eval()
            with torch.no_grad():
                tr = (probe(x_tr).argmax(-1) == y_tr).float().mean().item()
                te = (probe(x_te).argmax(-1) == y_te).float().mean().item()
            if loss.item() < best["loss"]:
                best = {"train_acc": tr, "test_acc": te, "loss": loss.item()}
    return best


def lin_regress(x_tr, y_tr, x_te, y_te, epochs, lr, wd, device):
    reg = nn.Linear(x_tr.shape[1], y_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(reg.parameters(), lr=lr, weight_decay=wd)
    x_tr, y_tr, x_te, y_te = (t.to(device) for t in (x_tr, y_tr, x_te, y_te))
    best = {"test_mse": float("inf"), "r2": float("nan")}
    for ep in range(epochs):
        reg.train()
        loss = F.mse_loss(reg(x_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            reg.eval()
            with torch.no_grad():
                pred = reg(x_te)
                te = F.mse_loss(pred, y_te).item()
                ss_res = ((pred - y_te) ** 2).sum().item()
                ss_tot = ((y_te - y_te.mean(0, keepdim=True)) ** 2).sum().item()
            if te < best["test_mse"]:
                best = {"test_mse": te, "r2": 1.0 - ss_res / max(ss_tot, 1e-9)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gt", default="outputs/trajectory_gt_val.pt")
    ap.add_argument("--max_videos", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe_split_seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--probe_epochs", type=int, default=2000)
    ap.add_argument("--probe_lr", type=float, default=1e-2)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt = torch.load(args.gt, map_location="cpu", weights_only=False)["per_video"]
    gt = {int(k): v for k, v in gt.items()}
    print(f"[gt] {len(gt)} videos with trajectory GT")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    d_static = int(a.get("d_static", 64)); d_dyn = int(a.get("d_dyn", 64))
    enc_hid = int(a.get("enc_hidden_ch", a.get("hidden_ch", 128)))
    d_event = int(a.get("d_event", 4))
    shared_trunk = bool(a.get("shared_trunk", False))
    use_ln = "norm_static.weight" in ckpt["encoder"]
    enc_unfrozen = bool(a.get("unfreeze_vae_enc", False))

    enc = LatentEncoder3D(d_static=d_static, d_dyn=d_dyn, hidden_ch=enc_hid,
                          shared_trunk=shared_trunk, use_layer_norm=use_ln).to(device)
    enc.load_state_dict(ckpt["encoder"]); enc.eval()
    eh = EventHead(d_dyn=d_dyn, d_event=d_event).to(device)
    if "event_head" in ckpt:
        eh.load_state_dict(ckpt["event_head"])
    eh.eval()
    print(f"[model] {args.ckpt}")
    print(f"[model] d_static={d_static} d_dyn={d_dyn} enc_h={enc_hid} use_ln={use_ln} "
          f"d_event={d_event} enc_unfrozen={enc_unfrozen}")
    if enc_unfrozen:
        print("[WARN] encoder-unfrozen run probed on CACHED latents -> lower bound "
              "(small encoder trained on re-encoded latents)")

    ds = ClevrerChunkPairs(args.cache_dir, split="val", val_frac=args.val_frac,
                           seed=args.seed, max_videos=args.max_videos)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=chunk_collate)
    t0 = time.time()
    per_chunk, per_video = collect(enc, eh, loader, gt, device)
    print(f"[encode] {len(per_video)} videos, {len(per_chunk)} (video,chunk) in {time.time()-t0:.1f}s")

    # 50/50 split by video
    vids = sorted(per_video.keys())
    random.Random(args.probe_split_seed).shuffle(vids)
    half = len(vids) // 2
    tr_v, te_v = set(vids[:half]), set(vids[half:])
    print(f"[probe] {len(tr_v)} train vids / {len(te_v)} test vids\n")

    pe, plr, pwd = args.probe_epochs, args.probe_lr, args.probe_wd
    results = {"identity": {}, "motion": {}, "collision": {}}

    # ---------- identity (per-video) ----------
    print("=== IDENTITY (modal attribute classification, per video) ===")
    print(f"{'feature':<12} {'group':<9} {'classes':>7} {'chance':>7} {'major':>7} "
          f"{'train':>7} {'TEST':>7} {'Δmaj':>7}")
    for feat in ("z_static", "z_dyn_last"):
        results["identity"][feat] = {}
        for name, lo, hi in ATTR_GROUPS:
            def build(vset):
                xs, ys = [], []
                for v in vset:
                    y = modal_label(per_video[v]["attrs"], per_video[v]["slot_mask"], lo, hi)
                    if y < 0: continue
                    xs.append(per_video[v][feat]); ys.append(y)
                return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
            x_tr, y_tr = build(tr_v); x_te, y_te = build(te_v)
            n_cls = hi - lo
            major = torch.bincount(y_te, minlength=n_cls).max().item() / max(y_te.numel(), 1)
            best = lin_classify(x_tr, y_tr, x_te, y_te, n_cls, pe, plr, pwd, device)
            results["identity"][feat][name] = {
                "test_acc": best["test_acc"], "train_acc": best["train_acc"],
                "majority": major, "chance": 1.0 / n_cls, "delta_maj": best["test_acc"] - major}
            print(f"{feat:<12} {name:<9} {n_cls:>7} {1.0/n_cls:>7.3f} {major:>7.3f} "
                  f"{best['train_acc']:>7.3f} {best['test_acc']:>7.3f} {best['test_acc']-major:>+7.3f}")

    # ---------- motion (per video×chunk) ----------
    print("\n=== MOTION (linear regression R², per video×chunk) ===")
    print(f"{'feature':<12} {'target':<8} {'test MSE':>10} {'R²':>8}")
    def stack_motion(tgt, vset):
        xs_s, xs_d, ys = [], [], []
        for (vid, ci), d in per_chunk.items():
            if vid not in vset: continue
            xs_s.append(d["z_static"]); xs_d.append(d["z_dyn_last"]); ys.append(d[tgt])
        return torch.stack(xs_s), torch.stack(xs_d), torch.stack(ys).float()
    for tgt in ("pos", "vel"):
        s_tr, d_tr, y_tr = stack_motion(tgt, tr_v)
        s_te, d_te, y_te = stack_motion(tgt, te_v)
        for feat, x_tr, x_te in (("z_static", s_tr, s_te), ("z_dyn_last", d_tr, d_te)):
            best = lin_regress(x_tr, y_tr, x_te, y_te, pe, plr, pwd, device)
            results["motion"].setdefault(feat, {})[tgt] = {"r2": best["r2"], "test_mse": best["test_mse"]}
            print(f"{feat:<12} {tgt:<8} {best['test_mse']:>10.5f} {best['r2']:>8.3f}")

    # ---------- collision / event (per video×chunk) ----------
    print("\n=== COLLISION (binary classification, per video×chunk) ===")
    print(f"{'feature':<12} {'major':>7} {'train':>7} {'TEST':>7} {'Δmaj':>7}")
    have_event = next(iter(per_chunk.values()))["z_event"] is not None
    feats = ["z_static", "z_dyn_last"] + (["z_event"] if have_event else [])
    def stack_coll(feat, vset):
        xs, ys = [], []
        for (vid, ci), d in per_chunk.items():
            if vid not in vset: continue
            xs.append(d[feat]); ys.append(d["collision"])
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
    for feat in feats:
        x_tr, y_tr = stack_coll(feat, tr_v); x_te, y_te = stack_coll(feat, te_v)
        major = torch.bincount(y_te, minlength=2).max().item() / max(y_te.numel(), 1)
        best = lin_classify(x_tr, y_tr, x_te, y_te, 2, pe, plr, pwd, device)
        results["collision"][feat] = {
            "test_acc": best["test_acc"], "train_acc": best["train_acc"],
            "majority": major, "delta_maj": best["test_acc"] - major}
        print(f"{feat:<12} {major:>7.3f} {best['train_acc']:>7.3f} {best['test_acc']:>7.3f} "
              f"{best['test_acc']-major:>+7.3f}")

    out_path = Path(args.out_json) if args.out_json else Path(args.ckpt).parent / "probe_v512_disentangle.json"
    out_path.write_text(json.dumps({
        "ckpt": args.ckpt, "enc_unfrozen_caveat": enc_unfrozen,
        "n_videos": len(per_video), "n_chunks": len(per_chunk),
        "split": {"max_videos": args.max_videos, "val_frac": args.val_frac, "seed": args.seed},
        "results": results}, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
