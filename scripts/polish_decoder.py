"""Decoder polish: freeze the factorized encoder, train a LARGER decoder to invert
the compact code better -> higher reconstruction fidelity (attacks the blur) WITHOUT
touching the disentangled code, so the z_static/z_dyn contribution is preserved.

Diagnosis (blur report + rate sweep): the recon gap to the Wan-VAE ceiling is our
small decoder's fault, not the input or the rate. So: keep the encoder frozen,
train a beefier decoder on latent-reconstruction MSE and report val-MSE vs the
shipped decoder. Lower latent-MSE -> the frozen VAE decodes a better latent -> higher
pixel PSNR.

Usage:
  python scripts/polish_decoder.py --ckpt .../v5_best.pt \
     --cache_dir .../wan_10000vid_W33 --hidden_ch 256 --depth 6 --steps 3000
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probes.clevrer_decode_baselines import build_our_encoder
from src.model.latent_decoder import SpatialGridDecoder
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from torch.utils.data import DataLoader


def build_decoder(a, hidden_ch, depth, device):
    def g(k, d):
        return a[k] if k in a else d
    return SpatialGridDecoder(
        d_static=int(g("d_static", 96)), static_grid=int(g("static_grid", 4) or 4),
        d_dyn=int(g("d_dyn", 256)), hidden_ch=int(hidden_ch),
        chunk_size_lat=int(g("chunk_size_lat", 9)), depth=int(depth), d_pose=0,
        dyn_spatial=bool(g("dyn_spatial", False)), dyn_grid=int(g("dyn_grid", 8))).to(device)


@torch.no_grad()
def val_mse(enc, dec, dl, device, max_batches=60):
    dec.eval(); tot = 0.0; n = 0
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        x = b["chunk_obs"].to(device); o = enc(x)
        recon = dec(o["z_static_grid"], o["z_dyn"])
        tot += F.mse_loss(recon, x).item() * len(x); n += len(x)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--hidden_ch", type=int, default=256, help="polished decoder width")
    ap.add_argument("--depth", type=int, default=6, help="polished decoder conv depth")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_videos", type=int, default=1500)
    ap.add_argument("--unfreeze_encoder", action="store_true",
                    help="also train the encoder (tests/fixes the ENCODER bottleneck: "
                         "can recon beat the frozen-encoder floor if the code can move?)")
    ap.add_argument("--enc_lr", type=float, default=1e-4, help="encoder LR when unfrozen")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)
    enc = build_our_encoder(a, ck["encoder"], device)
    if args.unfreeze_encoder:
        for p in enc.parameters():
            p.requires_grad_(True)
        print("[fix] encoder UNFROZEN -- training encoder+decoder jointly on recon", flush=True)

    mv = args.max_videos if args.max_videos > 0 else 100000
    tr = ClevrerChunkPairs(args.cache_dir, split="train", val_frac=float(a.get("val_frac", 0.1)),
                           seed=int(a.get("seed", 42)), max_videos=mv)
    va = ClevrerChunkPairs(args.cache_dir, split="val", val_frac=float(a.get("val_frac", 0.1)),
                           seed=int(a.get("seed", 42)), max_videos=mv)
    dtr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                     collate_fn=chunk_collate, drop_last=True)
    dva = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4,
                     collate_fn=chunk_collate)

    # baseline = the shipped decoder, as-is
    orig = build_decoder(a, int(a.get("dec_hidden_ch", 64)), int(a.get("dec_depth", 3)), device)
    orig_ok = True
    try:
        orig.load_state_dict(ck["decoder"])
    except Exception as e:
        print(f"[warn] shipped decoder load failed ({type(e).__name__}); baseline skipped")
        orig_ok = False
    base = val_mse(enc, orig, dva, device) if orig_ok else float("nan")
    print(f"[baseline] shipped decoder (h={a.get('dec_hidden_ch', 64)}, "
          f"depth={a.get('dec_depth', 3)}) val_mse={base:.5f}", flush=True)

    dec = build_decoder(a, args.hidden_ch, args.depth, device)
    nparam = sum(p.numel() for p in dec.parameters()) / 1e6
    orig_param = sum(p.numel() for p in orig.parameters()) / 1e6 if orig_ok else 0.0
    print(f"[polish] new decoder h={args.hidden_ch} depth={args.depth} "
          f"params={nparam:.2f}M (shipped {orig_param:.2f}M)", flush=True)
    param_groups = [{"params": dec.parameters(), "lr": args.lr}]
    if args.unfreeze_encoder:
        param_groups.append({"params": enc.parameters(), "lr": args.enc_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    it = iter(dtr); t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dtr); b = next(it)
        x = b["chunk_obs"].to(device)
        dec.train()
        if args.unfreeze_encoder:
            enc.train()
            o = enc(x)
        else:
            with torch.no_grad():
                o = enc(x)
        recon = dec(o["z_static_grid"], o["z_dyn"])
        loss = F.mse_loss(recon, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == args.steps:
            vm = val_mse(enc, dec, dva, device)
            impr = f"({100*(base-vm)/base:+.1f}% vs shipped)" if orig_ok else ""
            print(f"[step {step:>5}] train_mse={loss.item():.5f} val_mse={vm:.5f} "
                  f"{impr} ({time.time()-t0:.0f}s)", flush=True)
    final = val_mse(enc, dec, dva, device)
    print(f"[result] shipped_val_mse={base:.5f}  polished_val_mse={final:.5f}  "
          f"improvement={100*(base-final)/base:+.1f}%" if orig_ok else
          f"[result] polished_val_mse={final:.5f}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"decoder": dec.state_dict(), "args": a,
                    "hidden_ch": args.hidden_ch, "depth": args.depth}, args.out)
        print(f"[saved] {args.out}", flush=True)
    print("POLISH_DONE", flush=True)


if __name__ == "__main__":
    main()
