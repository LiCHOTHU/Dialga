from contextlib import contextmanager
import os
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision.utils as vutils

from src.data.clevrer_dataset import ClevrerTripletDataset
from src.model import WanFrozenEncoder, DiTLagrangian

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


@contextmanager
def _temporary_eval_mode(*modules):
    managed_modules = [module for module in modules if module is not None]
    previous_training_modes = [module.training for module in managed_modules]

    for module in managed_modules:
        module.eval()

    try:
        yield
    finally:
        for module, was_training in zip(managed_modules, previous_training_modes):
            module.train(was_training)


def calculate_del_residual(
    lagrangian,
    q_prev,
    q_curr,
    q_next,
    action_prev=None,
    action_curr=None,
):
    """Computes D2 Ld(q_{t-1}, q_t) + D1 Ld(q_t, q_{t+1})."""
    if not q_curr.requires_grad:
        q_curr = q_curr.detach().requires_grad_(True)

    l_prev = lagrangian(q_prev, q_curr, action=action_prev)
    l_curr = lagrangian(q_curr, q_next, action=action_curr)

    d2_prev = torch.autograd.grad(
        outputs=l_prev.sum(), inputs=q_curr, create_graph=True
    )[0]
    d1_curr = torch.autograd.grad(
        outputs=l_curr.sum(), inputs=q_curr, create_graph=True
    )[0]

    return d2_prev + d1_curr


def forward_dynamic_step(lagrangian, q_prev, q_curr, alpha=0.1, solver_steps=5):
    """
    Inference-only DEL solver used for rollout visualization.

    Training no longer backpropagates through this solver, so we avoid
    building higher-order graphs here. The inner solver still needs first-order
    gradients with respect to the candidate future state, so it re-enables
    autograd locally even when called from a no_grad visualization context.
    """
    with torch.enable_grad():
        q_curr_solver = q_curr.detach().requires_grad_(True)
        q_next_guess = (2 * q_curr - q_prev).detach().requires_grad_(True)
        l_prev = lagrangian(q_prev, q_curr_solver)
        d2_prev = torch.autograd.grad(
            outputs=l_prev.sum(), inputs=q_curr_solver, create_graph=False
        )[0]

        for _ in range(solver_steps):
            l_curr = lagrangian(q_curr_solver, q_next_guess)
            d1_curr = torch.autograd.grad(
                outputs=l_curr.sum(), inputs=q_curr_solver, create_graph=True
            )[0]
            residual = d2_prev + d1_curr
            residual_energy = 0.5 * residual.flatten(1).pow(2).sum(dim=1)
            solver_direction = torch.autograd.grad(
                outputs=residual_energy,
                inputs=q_next_guess,
                grad_outputs=torch.ones_like(residual_energy),
                create_graph=False,
            )[0]
            q_next_guess = (q_next_guess - alpha * solver_direction).detach().requires_grad_(True)

    return q_next_guess.detach()


def training_dynamic_step(
    lagrangian,
    q_prev,
    q_curr,
    alpha=0.01,
    solver_steps=1,
    detach_inputs=True,
):
    """
    Differentiable DEL solver used during training.

    Unlike the inference helper, this version keeps the graph so the
    Lagrangian receives gradient signal through the solver update.
    """
    q_prev_solver = q_prev.detach() if detach_inputs else q_prev
    q_curr_base = q_curr.detach() if detach_inputs else q_curr
    q_curr_solver = (
        q_curr_base.requires_grad_(True)
        if not q_curr_base.requires_grad
        else q_curr_base
    )
    q_next_guess = (2 * q_curr_solver - q_prev_solver)

    l_prev = lagrangian(q_prev_solver, q_curr_solver)
    d2_prev = torch.autograd.grad(
        outputs=l_prev.sum(), inputs=q_curr_solver, create_graph=True
    )[0]

    for _ in range(max(int(solver_steps), 1)):
        l_curr = lagrangian(q_curr_solver, q_next_guess)
        d1_curr = torch.autograd.grad(
            outputs=l_curr.sum(), inputs=q_curr_solver, create_graph=True
        )[0]

        residual = d2_prev + d1_curr
        residual_energy = 0.5 * residual.flatten(1).pow(2).sum(dim=1)

        solver_direction = torch.autograd.grad(
            outputs=residual_energy.sum(),
            inputs=q_next_guess,
            create_graph=True,
        )[0]
        q_next_guess = q_next_guess - alpha * solver_direction

    return q_next_guess


def compute_autoregressive_overfit_losses(
    lagrangian,
    q_sequence,
    alpha=0.01,
    solver_steps=1,
    rollout_steps=0,
    detach_between_steps=True,
    include_del=True,
):
    """
    Trains on a full video trajectory by rolling out autoregressively from the
    first two ground-truth latent states.
    """
    if q_sequence.shape[0] < 3:
        raise RuntimeError("Need at least 3 latent frames for autoregressive training.")

    max_rollout_steps = q_sequence.shape[0] - 2
    if rollout_steps <= 0:
        rollout_steps = max_rollout_steps
    rollout_steps = min(int(rollout_steps), max_rollout_steps)

    q_prev = q_sequence[0:1]
    q_curr = q_sequence[1:2]
    mse_sum = q_sequence.new_zeros(())
    del_sum = q_sequence.new_zeros(())

    for step_idx in range(rollout_steps):
        q_next_true = q_sequence[step_idx + 2 : step_idx + 3]
        q_next_pred = training_dynamic_step(
            lagrangian=lagrangian,
            q_prev=q_prev,
            q_curr=q_curr,
            alpha=alpha,
            solver_steps=solver_steps,
            detach_inputs=detach_between_steps,
        )

        mse_sum = mse_sum + F.mse_loss(q_next_pred, q_next_true)
        if include_del:
            residual_pred = calculate_del_residual(
                lagrangian=lagrangian,
                q_prev=q_prev,
                q_curr=q_curr,
                q_next=q_next_pred,
            )
            del_sum = del_sum + residual_pred.pow(2).mean()

        if detach_between_steps:
            q_prev = q_curr.detach()
            q_curr = q_next_pred.detach()
        else:
            q_prev = q_curr
            q_curr = q_next_pred

    normalizer = float(rollout_steps)
    return mse_sum / normalizer, del_sum / normalizer


def compute_ground_truth_sequence_losses(lagrangian, q_sequence, rollout_steps=0):
    """
    DEL-only sequence objective on ground-truth triples.

    The returned MSE is a constant-velocity anchor diagnostic only; it is not
    backpropagated through a differentiable DEL solver.
    """
    if q_sequence.shape[0] < 3:
        raise RuntimeError("Need at least 3 latent frames for sequence DEL training.")

    max_steps = q_sequence.shape[0] - 2
    if rollout_steps <= 0:
        rollout_steps = max_steps
    rollout_steps = min(int(rollout_steps), max_steps)

    anchor_mse_sum = q_sequence.new_zeros(())
    del_sum = q_sequence.new_zeros(())

    for step_idx in range(rollout_steps):
        q_prev = q_sequence[step_idx : step_idx + 1]
        q_curr = q_sequence[step_idx + 1 : step_idx + 2]
        q_next_true = q_sequence[step_idx + 2 : step_idx + 3]

        anchor_pred = 2 * q_curr - q_prev
        anchor_mse_sum = anchor_mse_sum + F.mse_loss(anchor_pred, q_next_true)

        residual_true = calculate_del_residual(
            lagrangian=lagrangian,
            q_prev=q_prev,
            q_curr=q_curr,
            q_next=q_next_true,
        )
        del_sum = del_sum + residual_true.pow(2).mean()

    normalizer = float(rollout_steps)
    return anchor_mse_sum / normalizer, del_sum / normalizer


def summarize_lagrangian_components(lagrangian, q_prev, q_curr):
    with torch.no_grad():
        components = lagrangian.compute_components(q_prev, q_curr)
    return {
        "mass_mean": components["mass"].mean().item(),
        "mass_min": components["mass"].min().item(),
        "mass_max": components["mass"].max().item(),
        "kinetic_mean": components["kinetic"].mean().item(),
        "potential_mean": components["potential"].mean().item(),
        "energy_mean": components["mechanical_energy"].mean().item(),
    }


def summarize_energy_drift(lagrangian, latent_sequence):
    if latent_sequence.shape[0] < 2:
        return None
    with torch.no_grad():
        components = lagrangian.compute_components(
            latent_sequence[:-1],
            latent_sequence[1:],
        )
        energy = components["mechanical_energy"]
        initial = energy[:1]
        relative_drift = (energy - initial).abs() / initial.abs().clamp_min(1e-8)
    return {
        "rollout_energy_drift": relative_drift.mean().item(),
        "rollout_energy_span": (energy.max() - energy.min()).item(),
    }


def infer_latent_shape(encoder, dataset, device):
    sample_prev, _, _ = dataset[0]
    sample_prev = sample_prev.unsqueeze(0).to(device)
    with _temporary_eval_mode(encoder), torch.no_grad():
        latent = encoder(sample_prev)
    return tuple(latent.shape[1:])


def evaluate_autoencoder_reconstruction(
    autoencoder,
    dataset,
    device,
    video_index=0,
    max_frames=4,
):
    if max_frames <= 0:
        return None

    frames = dataset.get_video_sequence(
        video_index=video_index,
        max_frames=max_frames,
    ).to(device)
    with _temporary_eval_mode(autoencoder), torch.no_grad():
        latents = autoencoder(frames)
        recon = autoencoder.decode(latents)

    if recon.shape != frames.shape:
        min_frames = min(recon.shape[0], frames.shape[0])
        log.warning(
            "VAE reconstruction shape mismatch: input=%s recon=%s. "
            "Comparing the first %d frames only.",
            tuple(frames.shape),
            tuple(recon.shape),
            min_frames,
        )
        frames = frames[:min_frames]
        recon = recon[:min_frames]

    return {
        "mse": F.mse_loss(recon, frames).item(),
        "mae": F.l1_loss(recon, frames).item(),
    }


def encode_video_sequence_in_chunks(encoder, frame_sequence, device, chunk_size):
    latents = []
    chunk_size = max(1, int(chunk_size))

    with _temporary_eval_mode(encoder), torch.no_grad():
        for start_idx in range(0, frame_sequence.shape[0], chunk_size):
            frame_chunk = frame_sequence[start_idx : start_idx + chunk_size].to(device)
            latents.append(encoder(frame_chunk))

    return torch.cat(latents, dim=0)


def _to_display_range(tensor):
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0)


def _save_video_artifact(video_frames, save_dir, stem, fps):
    mp4_path = save_dir / f"{stem}.mp4"
    try:
        from torchvision.io import write_video

        write_video(str(mp4_path), video_frames, fps=fps)
        return mp4_path
    except Exception as mp4_error:
        try:
            import imageio.v2 as imageio

            gif_path = save_dir / f"{stem}.gif"
            imageio.mimsave(
                str(gif_path),
                [frame.numpy() for frame in video_frames],
                fps=fps,
            )
            log.warning(
                "Fell back to GIF export because mp4 writing failed: %s",
                mp4_error,
            )
            return gif_path
        except Exception as gif_error:
            log.warning(
                "Could not save inference video. mp4 error: %s | gif error: %s",
                mp4_error,
                gif_error,
            )
            return None


def _repeat_video_frames(frame_batch, repeat_count):
    if repeat_count <= 1:
        return frame_batch
    return torch.cat([frame_batch[i : i + 1].repeat(repeat_count, 1, 1, 1) for i in range(frame_batch.shape[0])], dim=0)


def _rollout_latents(lagrangian, q_prev, q_curr, alpha, solver_steps, rollout_steps):
    rollout = []
    q_prev_roll = q_prev.detach()
    q_curr_roll = q_curr.detach()

    for _ in range(rollout_steps):
        q_next_roll = forward_dynamic_step(
            lagrangian=lagrangian,
            q_prev=q_prev_roll,
            q_curr=q_curr_roll,
            alpha=alpha,
            solver_steps=solver_steps,
        ).detach()
        rollout.append(q_next_roll)
        q_prev_roll, q_curr_roll = q_curr_roll, q_next_roll

    return rollout


def _uniform_sample_indices(length, sample_count):
    sample_count = max(1, min(int(sample_count), int(length)))
    return torch.linspace(0, length - 1, steps=sample_count).round().long()


def _init_wandb(cfg, output_dir):
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None

    if wandb is None:
        raise ImportError(
            "wandb logging is enabled but the `wandb` package is not installed. "
            "Install it in the training environment or set wandb.enabled=false."
        )

    run_name = wandb_cfg.get("name")
    if not run_name:
        run_name = output_dir.name

    run = wandb.init(
        project=wandb_cfg.get("project", "dialga"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train"),
        mode=wandb_cfg.get("mode", "online"),
        tags=list(wandb_cfg.get("tags", [])),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run.summary["hydra_output_dir"] = str(output_dir)
    return run


def run_inference_and_save(
    lagrangian,
    autoencoder,
    eval_dataset,
    epoch,
    cfg,
    device,
    save_dir,
    wandb_run=None,
):
    """
    Saves sparse-sampled GT vs predicted frames plus a side-by-side rollout video.
    """
    if autoencoder is None:
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    solver_alpha = float(cfg.training.get("solver_alpha", 0.1))
    solver_steps = int(cfg.training.get("solver_steps", 5))
    save_inference_images = bool(cfg.training.get("save_inference_images", True))
    save_inference_video = bool(cfg.training.get("save_inference_video", True))
    inference_sparse_frames = int(cfg.training.get("inference_sparse_frames", 8))
    inference_video_index = int(cfg.training.get("inference_video_index", -1))
    if inference_video_index < 0:
        inference_video_index = max(0, int(cfg.training.get("overfit_video_index", -1)))
    inference_video_max_frames = int(
        cfg.training.get(
            "inference_video_max_frames",
            int(cfg.dataset.get("video_num_frames", 128)),
        )
    )
    inference_video_fps = int(cfg.training.get("inference_video_fps", 4))
    inference_video_hold_frames = int(
        cfg.training.get("inference_video_hold_frames", 8)
    )

    gt_sequence = eval_dataset.get_video_sequence(
        video_index=inference_video_index,
        max_frames=inference_video_max_frames,
    )
    if gt_sequence.shape[0] < 2:
        raise RuntimeError(
            "Need at least 2 frames in the evaluation video sequence for rollout."
        )

    gt_sequence_display = _to_display_range(gt_sequence)

    with _temporary_eval_mode(lagrangian, autoencoder):
        with torch.no_grad():
            seed_frames = gt_sequence[:2].to(device)
            seed_latents = autoencoder(seed_frames)
            q_prev_sim = seed_latents[0:1]
            q_curr_sim = seed_latents[1:2]

            rollout_latents = _rollout_latents(
                lagrangian=lagrangian,
                q_prev=q_prev_sim,
                q_curr=q_curr_sim,
                alpha=solver_alpha,
                solver_steps=solver_steps,
                rollout_steps=max(gt_sequence.shape[0] - 2, 1),
            )
            if rollout_latents:
                future_latents = torch.cat(rollout_latents, dim=0)
                pred_latent_sequence = torch.cat(
                    [q_prev_sim, q_curr_sim, future_latents],
                    dim=0,
                )
                energy_metrics = summarize_energy_drift(lagrangian, pred_latent_sequence)
                decoded_future = autoencoder.decode(future_latents)
                pred_future_display = _to_display_range(decoded_future)
                pred_sequence_display = torch.cat(
                    [gt_sequence_display[:2], pred_future_display],
                    dim=0,
                )
            else:
                pred_latent_sequence = torch.cat([q_prev_sim, q_curr_sim], dim=0)
                energy_metrics = summarize_energy_drift(lagrangian, pred_latent_sequence)
                pred_sequence_display = gt_sequence_display[:2]
    pred_sequence_display = pred_sequence_display[: gt_sequence_display.shape[0]]
    sparse_indices = _uniform_sample_indices(
        gt_sequence_display.shape[0], inference_sparse_frames
    )

    if save_inference_images:
        comparison_grid = torch.cat(
            [gt_sequence_display[sparse_indices], pred_sequence_display[sparse_indices]],
            dim=0,
        )
        comparison_path = save_dir / f"epoch_{epoch:03d}_comparison.png"
        vutils.save_image(
            comparison_grid,
            str(comparison_path),
            nrow=sparse_indices.numel(),
            pad_value=1.0,
        )
        log.info(
            "Saved inference comparison strip to %s "
            "(top row: GT sparse samples | bottom row: predicted sparse samples)",
            comparison_path,
        )
    else:
        comparison_path = None

    if save_inference_video:
        side_by_side_sequence = torch.cat(
            [gt_sequence_display, pred_sequence_display],
            dim=-1,
        )
        side_by_side_sequence = _repeat_video_frames(
            side_by_side_sequence, max(inference_video_hold_frames, 1)
        )
        video_frames = (
            side_by_side_sequence.permute(0, 2, 3, 1).mul(255.0).round().to(torch.uint8)
        )
        video_path = _save_video_artifact(
            video_frames,
            save_dir,
            stem=f"epoch_{epoch:03d}_rollout",
            fps=inference_video_fps,
        )
        if video_path is not None:
            log.info(
                "Saved rollout video to %s "
                "(left: GT sequence | right: predicted sequence | %d frames | %d fps)",
                video_path,
                gt_sequence_display.shape[0],
                inference_video_fps,
            )
    else:
        video_path = None

    if energy_metrics is not None:
        log.info(
            "Rollout energy diagnostics | relative drift: %.6f | span: %.6f",
            energy_metrics["rollout_energy_drift"],
            energy_metrics["rollout_energy_span"],
        )

    if wandb_run is not None:
        media_payload = {}
        if comparison_path is not None:
            media_payload["media/comparison"] = wandb.Image(
                str(comparison_path),
                caption=f"Epoch {epoch}: top row GT sparse samples, bottom row predicted sparse samples",
            )
        if video_path is not None:
            media_payload["media/rollout"] = wandb.Video(
                str(video_path),
                caption=f"Epoch {epoch}: left GT sequence, right predicted sequence",
                fps=inference_video_fps,
            )
        if media_payload:
            wandb_run.log(media_payload, step=epoch)
        if energy_metrics is not None:
            wandb_run.log(
                {
                    "diagnostics/rollout_energy_drift": energy_metrics[
                        "rollout_energy_drift"
                    ],
                    "diagnostics/rollout_energy_span": energy_metrics[
                        "rollout_energy_span"
                    ],
                },
                step=epoch,
            )

    return comparison_path, video_path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    if int(cfg.training.batch_size) <= 0:
        raise ValueError("training.batch_size must be > 0.")
    if int(cfg.training.epochs) <= 0:
        raise ValueError("training.epochs must be > 0.")
    if int(cfg.training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers must be >= 0.")
    if int(cfg.training.get("overfit_num_workers", 0)) < 0:
        raise ValueError("training.overfit_num_workers must be >= 0.")

    log.info("Starting DIALGA training on %s...", device)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("Hydra run output directory: %s", output_dir)
    wandb_run = _init_wandb(cfg, output_dir)

    if not os.path.isfile(cfg.model.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {cfg.model.vae_ckpt}")

    # --- Load Data ---
    data_dir = (
        cfg.dataset.get("data_dir")
        or cfg.dataset.get("frames_dir")
        or cfg.dataset.get("videos_dir")
    )
    if data_dir is None:
        raise ValueError(
            "Please set cfg.dataset.data_dir, cfg.dataset.frames_dir, or "
            "cfg.dataset.videos_dir."
        )

    dataset = ClevrerTripletDataset(
        data_dir=data_dir,
        video_num_frames=int(cfg.dataset.get("video_num_frames", 128)),
    )
    base_dataset = dataset
    overfit_subset_size = int(cfg.training.get("overfit_subset_size", 0))
    overfit_video_index = int(cfg.training.get("overfit_video_index", -1))
    if overfit_video_index >= 0 and overfit_subset_size > 0:
        log.warning(
            "Both training.overfit_video_index=%d and training.overfit_subset_size=%d are set. "
            "Ignoring overfit_subset_size and using the selected video trajectory.",
            overfit_video_index,
            overfit_subset_size,
        )
    shuffle = True
    drop_last = True
    effective_batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.get("num_workers", 0))

    if overfit_video_index >= 0:
        video_indices = dataset.get_triplet_indices_for_video(overfit_video_index)
        dataset = Subset(dataset, video_indices)
        shuffle = bool(cfg.training.get("overfit_shuffle", False))
        drop_last = False
        effective_batch_size = min(effective_batch_size, len(video_indices))
        num_workers = int(cfg.training.get("overfit_num_workers", 0))
        log.info(
            "Video overfit mode enabled on video_index=%d with %d triplets | batch_size=%d | shuffle=%s",
            overfit_video_index,
            len(video_indices),
            effective_batch_size,
            shuffle,
        )
    elif overfit_subset_size > 0:
        subset_size = min(overfit_subset_size, len(dataset))
        dataset = Subset(dataset, list(range(subset_size)))
        shuffle = bool(cfg.training.get("overfit_shuffle", False))
        drop_last = False
        effective_batch_size = min(effective_batch_size, subset_size)
        num_workers = int(cfg.training.get("overfit_num_workers", 0))
        log.info(
            "Overfit mode enabled on first %d triplets | batch_size=%d | shuffle=%s",
            subset_size,
            effective_batch_size,
            shuffle,
        )

    eval_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if effective_batch_size <= 0:
        raise ValueError("Resolved training batch size must be > 0.")

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )

    # --- Initialize Models ---
    # Frozen WAN VAE used for both encode and decode to avoid a duplicate model on GPU.
    encoder = WanFrozenEncoder(vae_pth=cfg.model.vae_ckpt, device=device)
    log.warning(
        "WanFrozenEncoder is encoding individual frames with a video VAE. "
        "Use the reconstruction sanity check below to verify this latent is usable; "
        "object-centric CLEVRER coordinates remain the preferred physics state."
    )
    save_inference_images = bool(cfg.training.get("save_inference_images", True))
    save_inference_video = bool(cfg.training.get("save_inference_video", True))
    save_inference_visuals = save_inference_images or save_inference_video
    save_final_inference = bool(cfg.training.get("save_final_inference", True))

    actual_latent_shape = infer_latent_shape(encoder, dataset, device)
    configured_latent_shape = (
        cfg.model.latent_channels,
        cfg.model.latent_h,
        cfg.model.latent_w,
    )
    if actual_latent_shape != configured_latent_shape:
        log.warning(
            "Config latent shape %s does not match encoder output %s. "
            "Using encoder output for the DiT Lagrangian.",
            configured_latent_shape,
            actual_latent_shape,
        )

    reconstruction_check_frames = int(
        cfg.training.get("vae_reconstruction_check_frames", 4)
    )
    reconstruction_warn_mse = float(
        cfg.training.get("vae_reconstruction_warn_mse", 0.05)
    )
    reconstruction_metrics = evaluate_autoencoder_reconstruction(
        autoencoder=encoder,
        dataset=eval_dataset,
        device=device,
        video_index=max(overfit_video_index, 0),
        max_frames=reconstruction_check_frames,
    )
    if reconstruction_metrics is not None:
        log.info(
            "Single-frame VAE reconstruction check | MSE: %.6f | MAE: %.6f",
            reconstruction_metrics["mse"],
            reconstruction_metrics["mae"],
        )
        if reconstruction_metrics["mse"] > reconstruction_warn_mse:
            log.warning(
                "VAE reconstruction MSE %.6f exceeds warning threshold %.6f. "
                "The latent dynamics model may be fighting an off-distribution encoder.",
                reconstruction_metrics["mse"],
                reconstruction_warn_mse,
            )
        if wandb_run is not None:
            wandb_run.summary["diagnostics/vae_reconstruction_mse"] = (
                reconstruction_metrics["mse"]
            )
            wandb_run.summary["diagnostics/vae_reconstruction_mae"] = (
                reconstruction_metrics["mae"]
            )
    
    # The DiT Physics Engine
    lagrangian = DiTLagrangian(
        latent_channels=actual_latent_shape[0],
        latent_h=actual_latent_shape[1],
        latent_w=actual_latent_shape[2],
        patch_size=cfg.model.patch_size,
        hidden_size=cfg.model.hidden_size,
        depth=cfg.model.depth,
        num_heads=cfg.model.num_heads,
        action_dim=cfg.model.action_dim
    ).to(device)

    lambda_del = float(cfg.training.get("lambda_del", 1.0))
    lambda_solver_mse = float(cfg.training.get("lambda_solver_mse", 0.0))
    if lambda_del <= 0.0 and lambda_solver_mse <= 0.0:
        raise ValueError("At least one of lambda_del or lambda_solver_mse must be > 0.")
    if (overfit_video_index >= 0 or overfit_subset_size > 0) and lambda_solver_mse <= 0.0:
        log.warning(
            "Overfit mode is running with training.lambda_solver_mse=0.0. "
            "This fits only the DEL constraint on ground-truth latent triples; "
            "solver rollout accuracy is reported diagnostically but is not directly optimized. "
            "Set training.lambda_solver_mse>0 to test rollout memorization."
        )

    requested_gradient_checkpointing = bool(
        cfg.training.get("gradient_checkpointing", False)
    )
    if requested_gradient_checkpointing and (lambda_del > 0.0 or lambda_solver_mse > 0.0):
        log.warning(
            "Disabling gradient checkpointing because DEL/solver training uses "
            "higher-order autograd. Re-enable only for first-order ablations."
        )
        requested_gradient_checkpointing = False
    lagrangian.set_gradient_checkpointing(requested_gradient_checkpointing)

    optimizer = AdamW(
        lagrangian.parameters(), 
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay
    )

    grad_clip_norm = float(cfg.training.get("grad_clip_norm", 1.0))
    solver_alpha = float(cfg.training.get("solver_alpha", 0.1))
    training_solver_steps = int(cfg.training.get("training_solver_steps", 1))
    log_interval = int(cfg.training.get("log_interval", 10))
    diagnostics_every = int(cfg.training.get("diagnostics_every", log_interval))
    inference_every = int(cfg.training.get("inference_every", 1))
    if inference_every < 0:
        raise ValueError("training.inference_every must be >= 0.")
    solver_microbatch_size = max(
        1,
        int(cfg.training.get("solver_microbatch_size", effective_batch_size)),
    )
    autoregressive_rollout_steps = int(
        cfg.training.get("autoregressive_rollout_steps", 0)
    )
    autoregressive_detach_between_steps = bool(
        cfg.training.get("autoregressive_detach_between_steps", True)
    )
    sequence_encode_batch_size = max(
        1,
        int(cfg.training.get("sequence_encode_batch_size", 8)),
    )
    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("lr_warmup_epochs", 10))
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    if save_inference_visuals and inference_every == 0 and save_final_inference:
        log.info(
            "Periodic inference is disabled because training.inference_every=0, "
            "but final inference artifacts remain enabled via training.save_final_inference=true."
        )
    warmup_start_factor = 0.1
    if warmup_epochs == 0:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
        )
    elif total_epochs > warmup_epochs:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_epochs - warmup_epochs, 1),
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=max(warmup_epochs, 1),
        )

    # --- Training Loop ---
    try:
        last_inference_epoch = None
        for epoch in range(1, cfg.training.epochs + 1):
            lagrangian.train()
            epoch_anchor_mse_loss = 0.0
            epoch_solver_mse_loss = 0.0
            epoch_del_loss = 0.0
            epoch_diag_sums = {}
            epoch_diag_count = 0
            num_steps = 0

            if overfit_video_index >= 0:
                optimizer.zero_grad(set_to_none=True)
                overfit_sequence = base_dataset.get_video_sequence(
                    video_index=overfit_video_index,
                    max_frames=int(cfg.dataset.get("video_num_frames", 128)),
                )
                q_sequence = encode_video_sequence_in_chunks(
                    encoder=encoder,
                    frame_sequence=overfit_sequence,
                    device=device,
                    chunk_size=sequence_encode_batch_size,
                )

                loss_anchor_mse_tensor, loss_DEL_tensor = compute_ground_truth_sequence_losses(
                    lagrangian=lagrangian,
                    q_sequence=q_sequence,
                    rollout_steps=autoregressive_rollout_steps,
                )

                if lambda_solver_mse > 0.0:
                    loss_solver_mse_tensor, _ = compute_autoregressive_overfit_losses(
                        lagrangian=lagrangian,
                        q_sequence=q_sequence,
                        alpha=solver_alpha,
                        solver_steps=training_solver_steps,
                        rollout_steps=autoregressive_rollout_steps,
                        detach_between_steps=autoregressive_detach_between_steps,
                        include_del=False,
                    )
                else:
                    loss_solver_mse_tensor = q_sequence.new_zeros(())

                total_loss = (
                    lambda_del * loss_DEL_tensor
                    + lambda_solver_mse * loss_solver_mse_tensor
                )
                total_loss.backward()

                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        lagrangian.parameters(), max_norm=grad_clip_norm
                    )
                optimizer.step()

                epoch_anchor_mse_loss = loss_anchor_mse_tensor.item()
                epoch_solver_mse_loss = loss_solver_mse_tensor.item()
                epoch_del_loss = loss_DEL_tensor.item()
                component_stats = summarize_lagrangian_components(
                    lagrangian,
                    q_sequence[0:1],
                    q_sequence[1:2],
                )
                epoch_diag_sums.update(component_stats)
                epoch_diag_count = 1
                num_steps = 1
                log.info(
                    "Epoch %d overfit | anchor MSE: %.6f | solver MSE: %.6f | DEL: %.6f | LR: %.6e",
                    epoch,
                    epoch_anchor_mse_loss,
                    epoch_solver_mse_loss,
                    epoch_del_loss,
                    optimizer.param_groups[0]["lr"],
                )
            else:
                # Wrapped in tqdm for precise step tracking
                pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")

                for batch_idx, (o_prev, o_curr, o_next_true) in enumerate(pbar):
                    o_prev = o_prev.to(device)
                    o_curr = o_curr.to(device)
                    o_next_true = o_next_true.to(device)

                    # A: Encode strictly without tracking gradients
                    with torch.no_grad():
                        q_prev = encoder(o_prev)
                        q_curr = encoder(o_curr)
                        q_next_true = encoder(o_next_true)

                    # B: Build the higher-order graph on smaller latent chunks.
                    optimizer.zero_grad(set_to_none=True)
                    batch_size = q_prev.shape[0]
                    batch_anchor_mse_sum = 0.0
                    batch_solver_mse_sum = 0.0
                    batch_del_sum = 0.0

                    for start_idx in range(0, batch_size, solver_microbatch_size):
                        end_idx = min(start_idx + solver_microbatch_size, batch_size)
                        q_prev_mb = q_prev[start_idx:end_idx]
                        q_curr_mb = q_curr[start_idx:end_idx]
                        q_next_true_mb = q_next_true[start_idx:end_idx]

                        anchor_pred_mb = 2 * q_curr_mb - q_prev_mb
                        loss_anchor_mse_mb = F.mse_loss(
                            anchor_pred_mb,
                            q_next_true_mb,
                        )

                        if lambda_solver_mse > 0.0:
                            q_next_pred = training_dynamic_step(
                                lagrangian=lagrangian,
                                q_prev=q_prev_mb,
                                q_curr=q_curr_mb,
                                alpha=solver_alpha,
                                solver_steps=training_solver_steps,
                                detach_inputs=True,
                            )
                            loss_solver_mse_mb = F.mse_loss(
                                q_next_pred,
                                q_next_true_mb,
                            )
                        else:
                            loss_solver_mse_mb = q_prev_mb.new_zeros(())

                        residual_true_mb = calculate_del_residual(
                            lagrangian=lagrangian,
                            q_prev=q_prev_mb,
                            q_curr=q_curr_mb,
                            q_next=q_next_true_mb,
                        )
                        loss_DEL_mb = residual_true_mb.pow(2).mean()
                        total_loss_mb = (
                            lambda_del * loss_DEL_mb
                            + lambda_solver_mse * loss_solver_mse_mb
                        )
                        scaled_loss = total_loss_mb * (q_prev_mb.shape[0] / batch_size)
                        scaled_loss.backward()
                        batch_anchor_mse_sum += (
                            loss_anchor_mse_mb.item() * q_prev_mb.shape[0]
                        )
                        batch_solver_mse_sum += (
                            loss_solver_mse_mb.item() * q_prev_mb.shape[0]
                        )
                        batch_del_sum += loss_DEL_mb.item() * q_prev_mb.shape[0]

                        if (
                            diagnostics_every > 0
                            and batch_idx % diagnostics_every == 0
                            and start_idx == 0
                        ):
                            component_stats = summarize_lagrangian_components(
                                lagrangian,
                                q_prev_mb,
                                q_curr_mb,
                            )
                            for key, value in component_stats.items():
                                epoch_diag_sums[key] = (
                                    epoch_diag_sums.get(key, 0.0) + value
                                )
                            epoch_diag_count += 1

                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            lagrangian.parameters(), max_norm=grad_clip_norm
                        )
                    optimizer.step()

                    loss_anchor_mse = batch_anchor_mse_sum / batch_size
                    loss_solver_mse = batch_solver_mse_sum / batch_size
                    loss_DEL = batch_del_sum / batch_size
                    epoch_anchor_mse_loss += loss_anchor_mse
                    epoch_solver_mse_loss += loss_solver_mse
                    epoch_del_loss += loss_DEL
                    num_steps += 1

                    postfix = {
                        "AnchorMSE": f"{loss_anchor_mse:.4f}",
                        "DEL": f"{loss_DEL:.4f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                    if lambda_solver_mse > 0.0:
                        postfix["SolverMSE"] = f"{loss_solver_mse:.4f}"
                    if epoch_diag_count > 0:
                        postfix["Mass"] = (
                            f"{epoch_diag_sums['mass_mean'] / epoch_diag_count:.3f}"
                        )
                    pbar.set_postfix(postfix)

            if num_steps == 0:
                raise RuntimeError(
                    "The dataloader produced zero batches. Check that the dataset contains "
                    "enough frame triplets for the configured batch size."
                )

            epoch_anchor_mse_avg = epoch_anchor_mse_loss / num_steps
            epoch_solver_mse_avg = epoch_solver_mse_loss / num_steps
            epoch_del_avg = epoch_del_loss / num_steps
            del_to_anchor_ratio = epoch_del_avg / max(epoch_anchor_mse_avg, 1e-12)
            diag_avgs = {
                key: value / epoch_diag_count
                for key, value in epoch_diag_sums.items()
            } if epoch_diag_count > 0 else {}

            log.info(
                "Epoch %d complete | anchor MSE: %.6f | solver MSE: %.6f | "
                "DEL: %.6f | DEL/anchor: %.6e | LR: %.6e",
                epoch,
                epoch_anchor_mse_avg,
                epoch_solver_mse_avg,
                epoch_del_avg,
                del_to_anchor_ratio,
                optimizer.param_groups[0]["lr"],
            )
            if diag_avgs:
                log.info(
                    "Epoch %d diagnostics | mass mean/min/max: %.6f / %.6f / %.6f | "
                    "kinetic: %.6f | potential: %.6f | energy: %.6f",
                    epoch,
                    diag_avgs["mass_mean"],
                    diag_avgs["mass_min"],
                    diag_avgs["mass_max"],
                    diag_avgs["kinetic_mean"],
                    diag_avgs["potential_mean"],
                    diag_avgs["energy_mean"],
                )

            if wandb_run is not None:
                wandb_payload = {
                    "train/epoch": epoch,
                    "train/anchor_mse": epoch_anchor_mse_avg,
                    "train/solver_mse": epoch_solver_mse_avg,
                    "train/del": epoch_del_avg,
                    "train/del_to_anchor_mse": del_to_anchor_ratio,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for key, value in diag_avgs.items():
                    wandb_payload[f"diagnostics/{key}"] = value
                wandb_run.log(wandb_payload, step=epoch)

            scheduler.step()
                
            # Run inference and save a predicted frame at the end of the epoch
            if (
                save_inference_visuals
                and inference_every > 0
                and epoch % inference_every == 0
            ):
                run_inference_and_save(
                    lagrangian=lagrangian,
                    autoencoder=encoder,
                    eval_dataset=eval_dataset,
                    epoch=epoch,
                    cfg=cfg,
                    device=device,
                    save_dir=output_dir,
                    wandb_run=wandb_run,
                )
                last_inference_epoch = epoch

        if (
            save_inference_visuals
            and save_final_inference
            and last_inference_epoch != cfg.training.epochs
        ):
            log.info(
                "Saving final inference artifacts to %s after training.",
                output_dir,
            )
            run_inference_and_save(
                lagrangian=lagrangian,
                autoencoder=encoder,
                eval_dataset=eval_dataset,
                epoch=cfg.training.epochs,
                cfg=cfg,
                device=device,
                save_dir=output_dir,
                wandb_run=wandb_run,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == "__main__":
    main()
