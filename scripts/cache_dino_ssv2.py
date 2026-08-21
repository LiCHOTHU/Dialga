"""Cache DINOv2 patch features for SSv2, aligned to the SSv2 Wan-latent cache.

For every window in an SSv2 wan cache (metadata.json: video_id, start_frame), read
the clip, sample the 9 frames that anchor the 9 latent frames, run DINOv2, and pool
to an 8x8 patch grid. Output matches the CLEVRER DINO cache format so train_v5's
--lambda_mae semantic-distillation path can consume it (features.f16.bin + index.json,
shape [N, 9, 8, 8, 384]).

Usage:
  python scripts/cache_dino_ssv2.py \\
     --wan_cache_dir .../cache/ssv2_8000vid_W33 \\
     --video_root .../ssv2/videos_extracted/20bn-something-something-v2 \\
     --out_dir .../cache/dino_ssv2_8000vid_W33
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cache_wan_ssv2 import read_clip

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def dino_grid(model, frames, device, pool_hw=8):
    """frames (N,3,224,224) in [0,1] -> (N,pool_hw,pool_hw,D) float16."""
    x = ((frames - MEAN) / STD).to(device)
    out = model(x).last_hidden_state[:, 1:]                 # drop CLS -> (N, P, D)
    N, P, D = out.shape
    s = int(P ** 0.5)                                       # 16 for 224/14
    g = out.reshape(N, s, s, D).permute(0, 3, 1, 2)         # (N,D,s,s)
    g = F.adaptive_avg_pool2d(g, (pool_hw, pool_hw))        # (N,D,8,8)
    return g.permute(0, 2, 3, 1).half().cpu().numpy()       # (N,8,8,D)


def prep9(clip, start, W=33, T=9, size=224):
    idx = np.linspace(start, start + W - 1, T).round().astype(int)
    idx = np.clip(idx, 0, clip.shape[0] - 1)
    f = torch.from_numpy(clip[idx]).float().permute(0, 3, 1, 2) / 255.  # (9,3,H,Wd)
    _, _, H, Wd = f.shape; s = min(H, Wd)
    f = f[:, :, (H - s) // 2:(H - s) // 2 + s, (Wd - s) // 2:(Wd - s) // 2 + s]
    return F.interpolate(f, size=(size, size), mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wan_cache_dir", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--pool_hw", type=int, default=8)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    meta = json.loads((Path(args.wan_cache_dir) / "metadata.json").read_text())
    wins = meta["windows"]
    if args.max_windows:
        wins = wins[:args.max_windows]
    from transformers import AutoModel
    model = AutoModel.from_pretrained(args.model).to(dev).eval()
    D = model.config.hidden_size
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    N = len(wins)
    mm = np.memmap(out / "features.f16.bin", dtype=np.float16, mode="w+",
                   shape=(N, 9, args.pool_hw, args.pool_hw, D))
    root = Path(args.video_root)
    done = 0
    for i, w in enumerate(wins):
        clip = read_clip(str(root / f"{int(w['video_id'])}.webm"))
        if clip is None:
            mm[i] = 0; continue
        frames = prep9(clip, int(w["start_frame"]))
        mm[i] = dino_grid(model, frames, dev, args.pool_hw)
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{N}", flush=True)
    mm.flush()
    json.dump({"n_windows": N, "shape": [N, 9, args.pool_hw, args.pool_hw, D],
               "dtype": "float16", "wan_cache_dir": args.wan_cache_dir,
               "model": args.model, "pool_hw": args.pool_hw,
               "windows": [{"idx": i, "video_id": int(w["video_id"]),
                            "start_frame": int(w["start_frame"])} for i, w in enumerate(wins)]},
              open(out / "index.json", "w"))
    print(f"[done] {done}/{N} windows -> {out}")


if __name__ == "__main__":
    main()
