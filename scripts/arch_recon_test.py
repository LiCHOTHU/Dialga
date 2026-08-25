"""Fast isolation: does a FULL-RESOLUTION spatial static code (static_grid=8) improve
reconstruction over the coarse 4x4 static? Builds a FRESH encoder+decoder with the
given static_grid, trains pure-recon (isolates capacity, not the full loss suite),
reports val latent-MSE. Run for grid=4 and grid=8 and compare -- if 8 wins, the full
factorization-preserving retrain (clevrer_static8) is justified.

  python scripts/arch_recon_test.py --cache_dir ... --static_grid 4 --d_static 96
  python scripts/arch_recon_test.py --cache_dir ... --static_grid 8 --d_static 256
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.latent_encoder import LatentEncoder3D
from src.model.latent_decoder import SpatialGridDecoder
from src.data.clevrer_window import ClevrerChunkPairs, chunk_collate
from torch.utils.data import DataLoader


@torch.no_grad()
def val_mse(enc, dec, dl, device, max_batches=60):
    enc.eval(); dec.eval(); tot = 0.0; n = 0
    for i, b in enumerate(dl):
        if i >= max_batches:
            break
        x = b["chunk_obs"].to(device); o = enc(x)
        recon = dec(o["z_static_grid"], o["z_dyn"])
        tot += F.mse_loss(recon, x).item() * len(x); n += len(x)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--static_grid", type=int, default=4)
    ap.add_argument("--d_static", type=int, default=96)
    ap.add_argument("--d_dyn", type=int, default=256)
    ap.add_argument("--dyn_grid", type=int, default=8)
    ap.add_argument("--enc_hidden_ch", type=int, default=192)
    ap.add_argument("--dec_hidden_ch", type=int, default=384)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_videos", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    enc = LatentEncoder3D(
        d_static=args.d_static, d_dyn=args.d_dyn, hidden_ch=args.enc_hidden_ch,
        use_layer_norm=True, pool_type="spatial", static_grid=args.static_grid,
        chunk_size_lat=9, static_agg="conv", dyn_spatial=True, dyn_grid=args.dyn_grid,
        d_pose=0).to(device)
    dec = SpatialGridDecoder(
        d_static=args.d_static, static_grid=args.static_grid, d_dyn=args.d_dyn,
        hidden_ch=args.dec_hidden_ch, chunk_size_lat=9, depth=3, d_pose=0,
        dyn_spatial=True, dyn_grid=args.dyn_grid).to(device)
    npar = (sum(p.numel() for p in enc.parameters()) +
            sum(p.numel() for p in dec.parameters())) / 1e6
    c_static = args.d_static // (args.static_grid ** 2)
    print(f"[arch] static_grid={args.static_grid} d_static={args.d_static} "
          f"({c_static} ch/pos) d_dyn={args.d_dyn}  enc+dec={npar:.1f}M", flush=True)

    tr = ClevrerChunkPairs(args.cache_dir, split="train", val_frac=0.1, seed=42,
                           max_videos=args.max_videos)
    va = ClevrerChunkPairs(args.cache_dir, split="val", val_frac=0.1, seed=42,
                           max_videos=args.max_videos)
    dtr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4,
                     collate_fn=chunk_collate, drop_last=True)
    dva = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4,
                     collate_fn=chunk_collate)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    it = iter(dtr); t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dtr); b = next(it)
        x = b["chunk_obs"].to(device)
        enc.train(); dec.train()
        o = enc(x)
        recon = dec(o["z_static_grid"], o["z_dyn"])
        loss = F.mse_loss(recon, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps:
            vm = val_mse(enc, dec, dva, device)
            print(f"[grid{args.static_grid} step {step:>5}] train_mse={loss.item():.5f} "
                  f"val_mse={vm:.5f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[DONE grid{args.static_grid}] final_val_mse={val_mse(enc,dec,dva,device):.5f}", flush=True)
    print("ARCH_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
