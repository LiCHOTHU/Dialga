import logging
import random
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, random_split

from src.data.clevrer_sequence import ClevrerSequenceWindowDataset
from src.model import DiTLagrangian, LatentNextStatePredictor, LeWMPatchAutoencoder, SIGReg

try:
    import wandb
except ImportError:
    wandb = None

log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_is_trainable(module):
    return any(param.requires_grad for param in module.parameters())


def unwrap_dataset(dataset):
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current.base_dataset if hasattr(current, "base_dataset") else current


def encode_frame_sequence_batch(autoencoder, frames):
    batch_size, time_steps = frames.shape[:2]
    flat_frames = frames.flatten(0, 1)
    latent = autoencoder(flat_frames)
    return latent.view(batch_size, time_steps, *latent.shape[1:])


def compute_sigreg_loss(sigreg, latent_sequence):
    if sigreg is None:
        return latent_sequence.new_zeros(())
    proj = latent_sequence.transpose(0, 1).flatten(2)
    return sigreg(proj)


def calculate_del_residual(lagrangian, q_prev, q_curr, q_next):
    if not q_curr.requires_grad:
        q_curr = q_curr.detach().requires_grad_(True)

    l_prev = lagrangian(q_prev, q_curr)
    l_curr = lagrangian(q_curr, q_next)

    d2_prev = torch.autograd.grad(l_prev.sum(), q_curr, create_graph=True)[0]
    d1_curr = torch.autograd.grad(l_curr.sum(), q_curr, create_graph=True)[0]
    return d2_prev + d1_curr


def training_dynamic_step(lagrangian, q_prev, q_curr, alpha=0.1, solver_steps=1, detach_inputs=True):
    q_prev_solver = q_prev.detach() if detach_inputs else q_prev
    q_curr_solver = q_curr.detach() if detach_inputs else q_curr
    if not q_curr_solver.requires_grad:
        q_curr_solver = q_curr_solver.requires_grad_(True)
    q_next_guess = 2 * q_curr_solver - q_prev_solver

    l_prev = lagrangian(q_prev_solver, q_curr_solver)
    d2_prev = torch.autograd.grad(l_prev.sum(), q_curr_solver, create_graph=True)[0]

    for _ in range(max(int(solver_steps), 1)):
        l_curr = lagrangian(q_curr_solver, q_next_guess)
        d1_curr = torch.autograd.grad(l_curr.sum(), q_curr_solver, create_graph=True)[0]
        residual = d2_prev + d1_curr
        residual_energy = 0.5 * residual.flatten(1).pow(2).sum(dim=1)
        direction = torch.autograd.grad(residual_energy.sum(), q_next_guess, create_graph=True)[0]
        q_next_guess = q_next_guess - alpha * direction

    return q_next_guess


def compute_direct_predictor_sequence_losses(predictor, q_sequence):
    if q_sequence.shape[1] < 3:
        raise RuntimeError("Need at least 3 frames per sequence window.")

    teacher_terms = []
    rollout_terms = []
    q_prev_roll = q_sequence[:, 0]
    q_curr_roll = q_sequence[:, 1]
    for step_idx in range(q_sequence.shape[1] - 2):
        target = q_sequence[:, step_idx + 2]
        teacher_pred = predictor(q_sequence[:, step_idx], q_sequence[:, step_idx + 1])
        teacher_terms.append(F.mse_loss(teacher_pred, target))
        rollout_pred = predictor(q_prev_roll, q_curr_roll)
        rollout_terms.append(F.mse_loss(rollout_pred, target))
        q_prev_roll, q_curr_roll = q_curr_roll, rollout_pred

    teacher_loss = torch.stack(teacher_terms).mean()
    rollout_loss = torch.stack(rollout_terms).mean()
    anchor_loss = F.mse_loss(2 * q_sequence[:, 1:-1] - q_sequence[:, :-2], q_sequence[:, 2:])
    return anchor_loss, teacher_loss, rollout_loss


def compute_lagrangian_sequence_losses(lagrangian, q_sequence, alpha, solver_steps):
    if q_sequence.shape[1] < 3:
        raise RuntimeError("Need at least 3 frames per sequence window.")

    del_terms = []
    solver_terms = []
    q_prev_roll = q_sequence[:, 0]
    q_curr_roll = q_sequence[:, 1]
    for step_idx in range(q_sequence.shape[1] - 2):
        target = q_sequence[:, step_idx + 2]
        residual = calculate_del_residual(
            lagrangian,
            q_sequence[:, step_idx],
            q_sequence[:, step_idx + 1],
            target,
        )
        del_terms.append(residual.pow(2).mean())
        q_next_pred = training_dynamic_step(
            lagrangian=lagrangian,
            q_prev=q_prev_roll,
            q_curr=q_curr_roll,
            alpha=alpha,
            solver_steps=solver_steps,
            detach_inputs=True,
        )
        solver_terms.append(F.mse_loss(q_next_pred, target))
        q_prev_roll, q_curr_roll = q_curr_roll, q_next_pred

    del_loss = torch.stack(del_terms).mean()
    solver_loss = torch.stack(solver_terms).mean()
    anchor_loss = F.mse_loss(2 * q_sequence[:, 1:-1] - q_sequence[:, :-2], q_sequence[:, 2:])
    return anchor_loss, solver_loss, del_loss


@torch.no_grad()
def evaluate_validation(dynamics_model, dynamics_mode, autoencoder, val_loader, device, lambda_solver_mse, lambda_del, lambda_sigreg, sigreg, solver_alpha, solver_steps, max_batches):
    metric_sums = {
        "anchor_mse": 0.0,
        "solver_mse": 0.0,
        "del": 0.0,
        "sigreg": 0.0,
        "total": 0.0,
    }
    num_batches = 0
    dynamics_model.eval()
    autoencoder.eval()

    for batch_idx, frames in enumerate(val_loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        frames = frames.to(device)
        q_sequence = encode_frame_sequence_batch(autoencoder, frames)
        if dynamics_mode == "direct_predictor":
            anchor_loss, teacher_loss, rollout_loss = compute_direct_predictor_sequence_losses(dynamics_model, q_sequence)
            solver_loss = 0.5 * (teacher_loss + rollout_loss)
            del_loss = q_sequence.new_zeros(())
        else:
            anchor_loss, solver_loss, del_loss = compute_lagrangian_sequence_losses(
                lagrangian=dynamics_model,
                q_sequence=q_sequence,
                alpha=solver_alpha,
                solver_steps=solver_steps,
            )
        sigreg_loss = lambda_sigreg * compute_sigreg_loss(sigreg, q_sequence)
        total_loss = lambda_solver_mse * solver_loss + lambda_del * del_loss + sigreg_loss

        metric_sums["anchor_mse"] += anchor_loss.item()
        metric_sums["solver_mse"] += solver_loss.item()
        metric_sums["del"] += del_loss.item()
        metric_sums["sigreg"] += sigreg_loss.item()
        metric_sums["total"] += total_loss.item()
        num_batches += 1

    if num_batches == 0:
        return None
    return {key: value / num_batches for key, value in metric_sums.items()}


def save_checkpoint(output_dir, epoch, autoencoder, dynamics_model, optimizer, scheduler, cfg, metrics):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "autoencoder_state_dict": autoencoder.state_dict(),
            "dynamics_model_state_dict": dynamics_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _default_run_name(output_dir):
    job_name = Path(output_dir).name
    return job_name


def init_wandb(cfg, output_dir):
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled but not installed.")
    run = wandb.init(
        project=wandb_cfg.get("project", "dialga"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name") or _default_run_name(output_dir),
        group=wandb_cfg.get("group"),
        job_type=wandb_cfg.get("job_type", "train_state"),
        mode=wandb_cfg.get("mode", "online"),
        tags=list(wandb_cfg.get("tags", [])),
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run.summary["hydra_output_dir"] = str(output_dir)
    return run


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed = int(cfg.training.seed)
    set_seed(seed)

    requested_device = str(cfg.training.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("training.device is set to cuda, but CUDA is not available.")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    wandb_run = init_wandb(cfg, output_dir)

    if str(cfg.model.latent_source) != "lewm_patch":
        raise ValueError("This trainer only supports model.latent_source=lewm_patch.")

    data_dir = cfg.dataset.data_dir
    dataset = ClevrerSequenceWindowDataset(
        data_dir=data_dir,
        window_length=int(cfg.training.sequence_window_length),
        max_frames=int(cfg.dataset.video_num_frames),
        windows_per_video=int(cfg.training.sequence_windows_per_video),
        max_videos=int(cfg.training.sequence_max_videos),
        seed=seed,
    )

    train_subset_size = int(cfg.training.get("train_subset_size", 0))
    if train_subset_size > 0:
        subset_size = min(train_subset_size, len(dataset))
        subset_generator = torch.Generator().manual_seed(seed)
        subset_indices = torch.randperm(len(dataset), generator=subset_generator)[:subset_size].tolist()
        dataset = Subset(dataset, subset_indices)
        log.info("Training subset mode enabled with %d sampled windows.", subset_size)

    val_loader = None
    val_fraction = float(cfg.training.get("val_fraction", 0.0))
    if val_fraction > 0.0 and len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        split_generator = torch.Generator().manual_seed(seed)
        dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
        val_loader = DataLoader(
            val_dataset,
            batch_size=max(1, int(cfg.training.get("val_batch_size", cfg.training.batch_size))),
            shuffle=False,
            num_workers=int(cfg.training.get("val_num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            persistent_workers=int(cfg.training.get("val_num_workers", 0)) > 0,
        )
        log.info("Using held-out validation split | train samples: %d | val samples: %d", train_size, val_size)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.training.num_workers) > 0,
    )

    autoencoder = LeWMPatchAutoencoder(
        image_size=int(cfg.model.input_image_size),
        patch_size=int(cfg.model.lewm_patch_size),
        embed_dim=int(cfg.model.lewm_embed_dim),
        latent_channels=int(cfg.model.lewm_latent_channels),
        depth=int(cfg.model.lewm_encoder_depth),
        num_heads=int(cfg.model.lewm_num_heads),
        mlp_ratio=float(cfg.model.lewm_mlp_ratio),
    ).to(device)
    # Stage 1: learn the state and dynamics, keep the decoder frozen.
    autoencoder.decoder.requires_grad_(False)

    latent_shape = encode_frame_sequence_batch(autoencoder, dataset[0].unsqueeze(0).to(device)).shape[2:]
    log.info("LeWM encoder latent shape: %s", tuple(latent_shape))

    dynamics_mode = str(cfg.model.dynamics_model)
    if dynamics_mode == "direct_predictor":
        dynamics_model = LatentNextStatePredictor(
            latent_channels=latent_shape[0],
            hidden_channels=int(cfg.model.predictor_hidden_channels),
            num_blocks=int(cfg.model.predictor_num_blocks),
        ).to(device)
    elif dynamics_mode == "lagrangian":
        dynamics_model = DiTLagrangian(
            latent_channels=latent_shape[0],
            latent_h=latent_shape[1],
            latent_w=latent_shape[2],
            patch_size=int(cfg.model.patch_size),
            hidden_size=int(cfg.model.hidden_size),
            depth=int(cfg.model.depth),
            num_heads=int(cfg.model.num_heads),
            action_dim=int(cfg.model.action_dim),
        ).to(device)
        dynamics_model.set_gradient_checkpointing(False)
    else:
        raise ValueError("Unsupported model.dynamics_model. Use 'direct_predictor' or 'lagrangian'.")

    sigreg = None
    lambda_sigreg = float(cfg.training.get("lambda_sigreg", 0.0))
    if lambda_sigreg > 0.0:
        sigreg = SIGReg(
            knots=int(cfg.training.get("sigreg_knots", 17)),
            num_proj=int(cfg.training.get("sigreg_num_proj", 256)),
        ).to(device)

    optimizer = AdamW(
        list(autoencoder.parameters()) + list(dynamics_model.parameters()),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    use_scheduler = bool(cfg.training.get("use_scheduler", True))
    total_epochs = int(cfg.training.epochs)
    warmup_epochs = max(0, min(int(cfg.training.get("lr_warmup_epochs", 0)), total_epochs))
    if not use_scheduler:
        scheduler = None
    elif warmup_epochs == 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    elif total_epochs > warmup_epochs:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1)),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(warmup_epochs, 1))

    lambda_solver = float(cfg.training.lambda_solver_mse)
    lambda_del = float(cfg.training.lambda_del)
    solver_alpha = float(cfg.training.solver_alpha)
    solver_steps = int(cfg.training.training_solver_steps)
    grad_clip_norm = float(cfg.training.grad_clip_norm)
    log_interval = int(cfg.training.log_interval)
    val_every_steps = int(cfg.training.get("val_every_steps", 0))
    val_max_batches = int(cfg.training.get("val_max_batches", 0))
    checkpoint_every = int(cfg.training.get("checkpoint_every", 0))

    last_epoch_metrics = None
    global_step = 0
    for epoch in range(1, total_epochs + 1):
        autoencoder.train()
        autoencoder.decoder.eval()
        dynamics_model.train()

        epoch_anchor = 0.0
        epoch_solver = 0.0
        epoch_del = 0.0
        epoch_sigreg = 0.0
        epoch_total = 0.0
        epoch_diag_sums = {}
        epoch_diag_count = 0
        num_steps = 0

        for batch_idx, frames in enumerate(loader, start=1):
            frames = frames.to(device)
            optimizer.zero_grad(set_to_none=True)
            q_sequence = encode_frame_sequence_batch(autoencoder, frames)

            if dynamics_mode == "direct_predictor":
                anchor_loss, teacher_loss, rollout_loss = compute_direct_predictor_sequence_losses(dynamics_model, q_sequence)
                solver_loss = 0.5 * (teacher_loss + rollout_loss)
                del_loss = q_sequence.new_zeros(())
            else:
                anchor_loss, solver_loss, del_loss = compute_lagrangian_sequence_losses(
                    lagrangian=dynamics_model,
                    q_sequence=q_sequence,
                    alpha=solver_alpha,
                    solver_steps=solver_steps,
                )

            sigreg_loss = lambda_sigreg * compute_sigreg_loss(sigreg, q_sequence)
            total_loss = lambda_solver * solver_loss + lambda_del * del_loss + sigreg_loss
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(autoencoder.parameters()) + list(dynamics_model.parameters()), max_norm=grad_clip_norm)
            optimizer.step()

            global_step += 1
            num_steps += 1
            epoch_anchor += anchor_loss.item()
            epoch_solver += solver_loss.item()
            epoch_del += del_loss.item()
            epoch_sigreg += sigreg_loss.item()
            epoch_total += total_loss.item()

            if dynamics_mode == "lagrangian":
                comps = dynamics_model.compute_components(q_sequence[:, 0], q_sequence[:, 1])
                epoch_diag_sums.setdefault("mass_mean", 0.0)
                epoch_diag_sums.setdefault("mass_min", 0.0)
                epoch_diag_sums.setdefault("mass_max", 0.0)
                epoch_diag_sums.setdefault("kinetic_mean", 0.0)
                epoch_diag_sums.setdefault("potential_mean", 0.0)
                epoch_diag_sums.setdefault("energy_mean", 0.0)
                epoch_diag_sums["mass_mean"] += comps["mass"].mean().item()
                epoch_diag_sums["mass_min"] += comps["mass"].min().item()
                epoch_diag_sums["mass_max"] += comps["mass"].max().item()
                epoch_diag_sums["kinetic_mean"] += comps["kinetic"].mean().item()
                epoch_diag_sums["potential_mean"] += comps["potential"].mean().item()
                epoch_diag_sums["energy_mean"] += comps["mechanical_energy"].mean().item()
                epoch_diag_count += 1

            if batch_idx % log_interval == 0:
                log.info(
                    "Epoch %d step %d | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | DEL: %.6f | SIGReg: %.6f | LR: %.6e",
                    epoch,
                    batch_idx,
                    total_loss.item(),
                    anchor_loss.item(),
                    solver_loss.item(),
                    del_loss.item(),
                    sigreg_loss.item(),
                    optimizer.param_groups[0]["lr"],
                )

            if wandb_run is not None and batch_idx % log_interval == 0:
                wandb_run.log(
                    {
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/total": total_loss.item(),
                        "train/anchor_mse": anchor_loss.item(),
                        "train/solver_mse": solver_loss.item(),
                        "train/del": del_loss.item(),
                        "train/sigreg": sigreg_loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            if val_loader is not None and val_every_steps > 0 and global_step % val_every_steps == 0:
                val_metrics = evaluate_validation(
                    dynamics_model=dynamics_model,
                    dynamics_mode=dynamics_mode,
                    autoencoder=autoencoder,
                    val_loader=val_loader,
                    device=device,
                    lambda_solver_mse=lambda_solver,
                    lambda_del=lambda_del,
                    lambda_sigreg=lambda_sigreg,
                    sigreg=sigreg,
                    solver_alpha=solver_alpha,
                    solver_steps=solver_steps,
                    max_batches=val_max_batches,
                )
                if val_metrics is not None:
                    log.info(
                        "Val step %d | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | DEL: %.6f | SIGReg: %.6f",
                        global_step,
                        val_metrics["total"],
                        val_metrics["anchor_mse"],
                        val_metrics["solver_mse"],
                        val_metrics["del"],
                        val_metrics["sigreg"],
                    )
                    if wandb_run is not None:
                        wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

        epoch_metrics = {
            "train/total": epoch_total / max(num_steps, 1),
            "train/anchor_mse": epoch_anchor / max(num_steps, 1),
            "train/solver_mse": epoch_solver / max(num_steps, 1),
            "train/del": epoch_del / max(num_steps, 1),
            "train/sigreg": epoch_sigreg / max(num_steps, 1),
            "train/lr": optimizer.param_groups[0]["lr"],
        }
        if epoch_diag_count > 0:
            epoch_metrics.update({f"diagnostics/{k}": v / epoch_diag_count for k, v in epoch_diag_sums.items()})

        log.info(
            "Epoch %d complete | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | DEL: %.6f | SIGReg: %.6f | LR: %.6e",
            epoch,
            epoch_metrics["train/total"],
            epoch_metrics["train/anchor_mse"],
            epoch_metrics["train/solver_mse"],
            epoch_metrics["train/del"],
            epoch_metrics["train/sigreg"],
            epoch_metrics["train/lr"],
        )
        if epoch_diag_count > 0:
            log.info(
                "Epoch %d diagnostics | mass mean/min/max: %.6f / %.6f / %.6f | kinetic: %.6f | potential: %.6f | energy: %.6f",
                epoch,
                epoch_metrics["diagnostics/mass_mean"],
                epoch_metrics["diagnostics/mass_min"],
                epoch_metrics["diagnostics/mass_max"],
                epoch_metrics["diagnostics/kinetic_mean"],
                epoch_metrics["diagnostics/potential_mean"],
                epoch_metrics["diagnostics/energy_mean"],
            )

        if wandb_run is not None:
            wandb_run.log({"train/epoch": epoch, **epoch_metrics}, step=global_step)

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            checkpoint_path = save_checkpoint(output_dir, epoch, autoencoder, dynamics_model, optimizer, scheduler, cfg, epoch_metrics)
            log.info("Saved checkpoint to %s", checkpoint_path)

        if scheduler is not None:
            scheduler.step()
        last_epoch_metrics = epoch_metrics

    checkpoint_path = save_checkpoint(output_dir, total_epochs, autoencoder, dynamics_model, optimizer, scheduler, cfg, last_epoch_metrics)
    log.info("Saved final checkpoint to %s", checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
