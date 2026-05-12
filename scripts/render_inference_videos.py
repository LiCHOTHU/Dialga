"""Render decoder inference videos (GT / encoder-recon / rollout-recon) from
an existing stage2.pt checkpoint, without retraining.

Writes one decoder_video_<vid>.gif per sampled video into the output dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from omegaconf import OmegaConf

from src.data.clevrer_paired import ClevrerPairedDataset
from src.model.slot_lagrangian import SlotPixelDecoder
from scripts.train_slot import log_inference_videos
from scripts.eval_state_overlay import build_modules_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to stage2.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_videos", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt["config"]
    cfg = OmegaConf.create(cfg_dict)

    dataset = ClevrerPairedDataset(
        data_dir=str(cfg.dataset.data_dir),
        annotation_dir=str(cfg.dataset.annotation_dir),
        split=str(cfg.dataset.split),
        window_length=int(cfg.training.window_length),
        frames_per_video=int(cfg.dataset.video_num_frames),
        windows_per_video=int(cfg.training.windows_per_video),
        max_videos=int(cfg.training.max_videos),
        max_objects=int(cfg.dataset.max_objects),
        coordinate_mode=str(cfg.dataset.coordinate_mode),
        image_size=int(cfg.dataset.image_size),
        seed=int(cfg.training.seed),
    )

    cfg_dict, encoder, lagrangian, impulse, enc_type, dyn_type = build_modules_from_ckpt(
        ckpt, dataset.attr_dim, device,
    )

    d_static = int(cfg.model.get("d_static", 16))
    decoder = SlotPixelDecoder(
        num_state_dims=int(cfg.model.num_state_dims),
        attr_dim=d_static,
        max_objects=int(cfg.dataset.max_objects),
        image_size=int(cfg.dataset.image_size),
        hidden=int(cfg.model.decoder_hidden),
        slot_embed=int(cfg.model.decoder_slot_embed),
        num_blocks=int(cfg.model.decoder_blocks),
        grid_size=int(cfg.model.decoder_grid_size),
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"]); decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    log_inference_videos(
        cfg, device, encoder, lagrangian, impulse, decoder,
        dataset, wandb_run=None, num_videos=args.num_videos,
        output_dir=out_dir,
    )
    print(f"Wrote decoder inference videos to {out_dir}")


if __name__ == "__main__":
    main()
