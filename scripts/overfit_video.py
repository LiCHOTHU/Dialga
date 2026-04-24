import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.data.clevrer_dataset import ClevrerTripletDataset
from src.model.autoencoder import WanFrozenEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit the latent video stack to a single CLEVRER video.",
    )
    parser.add_argument(
        "--data-dir",
        default="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video",
        help="CLEVRER frame/video root.",
    )
    parser.add_argument(
        "--vae-ckpt",
        default="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        help="Path to the frozen WAN VAE checkpoint.",
    )
    parser.add_argument("--video-index", type=int, default=0, help="Video index to overfit.")
    parser.add_argument("--max-frames", type=int, default=128, help="Number of frames to use.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="AdamW weight decay.")
    parser.add_argument("--hidden-channels", type=int, default=256, help="Predictor width.")
    parser.add_argument("--num-blocks", type=int, default=4, help="Residual block count.")
    parser.add_argument("--step-dim", type=int, default=64, help="Step embedding width.")
    parser.add_argument("--rollout-weight", type=float, default=1.0, help="Autoregressive rollout loss weight.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=10, help="Epoch logging interval.")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint/visual save interval.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to ./output/video_overfit_<video-index>.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return x + hidden


class StepConditionedVideoPredictor(nn.Module):
    def __init__(self, latent_channels, max_steps, hidden_channels, num_blocks, step_dim):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps, step_dim)
        self.in_proj = nn.Conv2d(
            latent_channels * 2 + step_dim,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.out_norm = nn.GroupNorm(groups, hidden_channels)
        self.out_proj = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, q_prev, q_curr, step_idx):
        if step_idx.dim() == 0:
            step_idx = step_idx.unsqueeze(0)
        step_features = self.step_embed(step_idx).to(q_curr.dtype)
        step_features = step_features[:, :, None, None].expand(
            -1,
            -1,
            q_curr.shape[-2],
            q_curr.shape[-1],
        )
        hidden = torch.cat([q_prev, q_curr, step_features], dim=1)
        hidden = self.in_proj(hidden)
        hidden = self.blocks(hidden)
        return self.out_proj(F.silu(self.out_norm(hidden)))


def rollout_latents(model, seed_latents):
    predictions = [seed_latents[0:1], seed_latents[1:2]]
    q_prev = predictions[0]
    q_curr = predictions[1]

    for step in range(seed_latents.shape[0] - 2):
        step_idx = torch.tensor([step], device=seed_latents.device, dtype=torch.long)
        q_next = model(q_prev, q_curr, step_idx)
        predictions.append(q_next)
        q_prev, q_curr = q_curr, q_next

    return torch.cat(predictions, dim=0)


def compute_losses(model, latents):
    q_prev = latents[:-2]
    q_curr = latents[1:-1]
    q_next = latents[2:]
    step_ids = torch.arange(latents.shape[0] - 2, device=latents.device)

    teacher_forcing_pred = model(q_prev, q_curr, step_ids)
    teacher_forcing_loss = F.mse_loss(teacher_forcing_pred, q_next)

    rollout_loss = latents.new_zeros(())
    q_prev_roll = latents[0:1]
    q_curr_roll = latents[1:2]
    for step in range(latents.shape[0] - 2):
        step_idx = torch.tensor([step], device=latents.device, dtype=torch.long)
        q_next_pred = model(q_prev_roll, q_curr_roll, step_idx)
        rollout_loss = rollout_loss + F.mse_loss(q_next_pred, latents[step + 2 : step + 3])
        q_prev_roll, q_curr_roll = q_curr_roll, q_next_pred
    rollout_loss = rollout_loss / float(latents.shape[0] - 2)
    return teacher_forcing_loss, rollout_loss


@torch.no_grad()
def evaluate(model, autoencoder, latents, frames):
    predicted_latents = rollout_latents(model, latents)
    decoded_frames = autoencoder.decode(predicted_latents)
    latent_mse = F.mse_loss(predicted_latents, latents).item()
    frame_mse = F.mse_loss(decoded_frames, frames).item()
    frame_mae = F.l1_loss(decoded_frames, frames).item()
    psnr = -10.0 * math.log10(max(frame_mse, 1e-12))
    return {
        "predicted_latents": predicted_latents,
        "decoded_frames": decoded_frames,
        "latent_rollout_mse": latent_mse,
        "frame_mse": frame_mse,
        "frame_mae": frame_mae,
        "psnr": psnr,
    }


def to_display_range(tensor):
    tensor = tensor.detach().cpu()
    if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0)


def save_video_artifact(video_frames, output_dir, stem, fps):
    mp4_path = output_dir / f"{stem}.mp4"
    try:
        from torchvision.io import write_video

        write_video(str(mp4_path), video_frames, fps=fps)
        return mp4_path
    except Exception:
        try:
            import imageio.v2 as imageio

            gif_path = output_dir / f"{stem}.gif"
            imageio.mimsave(str(gif_path), [frame.numpy() for frame in video_frames], fps=fps)
            return gif_path
        except Exception:
            return None


def save_visuals(output_dir, epoch, gt_frames, pred_frames):
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_display = to_display_range(gt_frames)
    pred_display = to_display_range(pred_frames)

    sample_count = min(8, gt_display.shape[0])
    sample_indices = torch.linspace(0, gt_display.shape[0] - 1, steps=sample_count).round().long()
    comparison_grid = torch.cat([gt_display[sample_indices], pred_display[sample_indices]], dim=0)
    comparison_path = output_dir / f"epoch_{epoch:03d}_comparison.png"
    vutils.save_image(comparison_grid, str(comparison_path), nrow=sample_count, pad_value=1.0)

    side_by_side = torch.cat([gt_display, pred_display], dim=-1)
    video_frames = side_by_side.permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8)
    video_path = save_video_artifact(video_frames, output_dir, f"epoch_{epoch:03d}_rollout", fps=12)
    return comparison_path, video_path


def save_checkpoint(output_dir, name, model, optimizer, args, epoch, metrics):
    checkpoint_path = output_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main():
    args = parse_args()
    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir or f"output/video_overfit_{args.video_index:05d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ClevrerTripletDataset(
        data_dir=args.data_dir,
        video_num_frames=max(args.max_frames, 3),
    )
    frames = dataset.get_video_sequence(
        video_index=args.video_index,
        max_frames=args.max_frames,
    ).to(device)
    if frames.shape[0] < 3:
        raise RuntimeError("Need at least 3 frames to overfit the video predictor.")

    autoencoder = WanFrozenEncoder(vae_pth=args.vae_ckpt, device=device)
    with torch.no_grad():
        latents = autoencoder(frames)

    model = StepConditionedVideoPredictor(
        latent_channels=latents.shape[1],
        max_steps=latents.shape[0] - 2,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        step_dim=args.step_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = float("inf")
    best_summary = None

    run_config = {
        "video_index": args.video_index,
        "num_frames": int(frames.shape[0]),
        "latent_shape": list(latents.shape),
        "device": str(device),
        **vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Using device: {device}")
    print(f"Video index: {args.video_index} | frames: {frames.shape[0]} | latents: {tuple(latents.shape)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        teacher_forcing_loss, rollout_loss = compute_losses(model, latents)
        total_loss = teacher_forcing_loss + args.rollout_weight * rollout_loss
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        should_log = epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs
        should_save = epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs

        if should_log or should_save:
            model.eval()
            with torch.no_grad():
                eval_metrics = evaluate(model, autoencoder, latents, frames)

            summary = {
                "epoch": epoch,
                "train_total_loss": float(total_loss.item()),
                "train_teacher_forcing_loss": float(teacher_forcing_loss.item()),
                "train_rollout_loss": float(rollout_loss.item()),
                "latent_rollout_mse": eval_metrics["latent_rollout_mse"],
                "frame_mse": eval_metrics["frame_mse"],
                "frame_mae": eval_metrics["frame_mae"],
                "psnr": eval_metrics["psnr"],
            }

            if should_log:
                print(
                    "epoch={epoch:03d} total={total:.6f} tf={tf:.6f} rollout={roll:.6f} "
                    "latent_mse={latent:.6f} frame_mse={frame:.6f} psnr={psnr:.2f}".format(
                        epoch=epoch,
                        total=summary["train_total_loss"],
                        tf=summary["train_teacher_forcing_loss"],
                        roll=summary["train_rollout_loss"],
                        latent=summary["latent_rollout_mse"],
                        frame=summary["frame_mse"],
                        psnr=summary["psnr"],
                    )
                )

            if summary["frame_mse"] < best_metric:
                best_metric = summary["frame_mse"]
                best_summary = summary
                save_checkpoint(
                    output_dir=output_dir,
                    name="best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    metrics=summary,
                )

            if should_save:
                save_checkpoint(
                    output_dir=output_dir,
                    name=f"epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    metrics=summary,
                )
                save_visuals(
                    output_dir=output_dir,
                    epoch=epoch,
                    gt_frames=frames,
                    pred_frames=eval_metrics["decoded_frames"],
                )

    final_metrics = evaluate(model, autoencoder, latents, frames)
    final_summary = {
        "epoch": args.epochs,
        "latent_rollout_mse": final_metrics["latent_rollout_mse"],
        "frame_mse": final_metrics["frame_mse"],
        "frame_mae": final_metrics["frame_mae"],
        "psnr": final_metrics["psnr"],
        "best_frame_mse": best_metric,
        "best_epoch": None if best_summary is None else best_summary["epoch"],
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    save_checkpoint(
        output_dir=output_dir,
        name="final.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        epoch=args.epochs,
        metrics=final_summary,
    )
    save_visuals(
        output_dir=output_dir,
        epoch=args.epochs,
        gt_frames=frames,
        pred_frames=final_metrics["decoded_frames"],
    )

    print("Finished overfit run.")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
