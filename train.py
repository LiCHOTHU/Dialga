from contextlib import contextmanager
import logging
import os
from pathlib import Path
import random
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
import torchvision.utils as vutils

from src.data.clevrer_dataset import ClevrerTripletDataset
from src.data.clevrer_sequence import ClevrerSequenceWindowDataset
from src.model import (
    DiTLagrangian,
    DinoV2FrozenEncoder,
    FrozenDINOAutoencoder,
    LatentNextStatePredictor,
    LeWMPatchAutoencoder,
    ResidualStateProjector,
    SIGReg,
    WanFrozenEncoder,
)

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
            q_next_guess = (
                q_next_guess - alpha * solver_direction
            ).detach().requires_grad_(True)

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
    q_next_guess = 2 * q_curr_solver - q_prev_solver

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


def compute_batched_sequence_losses(
    lagrangian,
    q_sequence,
    alpha,
    solver_steps,
    detach_between_steps,
    autoencoder=None,
    target_frames=None,
    enable_decode_grad=False,
):
    if q_sequence.dim() != 5:
        raise ValueError(
            f"Expected batched latent sequence with shape (B, T, C, H, W), got {tuple(q_sequence.shape)}."
        )
    if q_sequence.shape[1] < 3:
        raise RuntimeError("Need at least 3 frames in each sequence window.")

    anchor_pred = 2 * q_sequence[:, 1:-1] - q_sequence[:, :-2]
    anchor_loss = F.mse_loss(anchor_pred, q_sequence[:, 2:])

    del_terms = []
    for step_idx in range(1, q_sequence.shape[1] - 1):
        residual = calculate_del_residual(
            lagrangian=lagrangian,
            q_prev=q_sequence[:, step_idx - 1],
            q_curr=q_sequence[:, step_idx],
            q_next=q_sequence[:, step_idx + 1],
        )
        del_terms.append(residual.pow(2).mean())
    del_loss = torch.stack(del_terms).mean()

    solver_terms = []
    pred_recon_terms = []
    q_prev_roll = q_sequence[:, 0]
    q_curr_roll = q_sequence[:, 1]
    for step_idx in range(q_sequence.shape[1] - 2):
        q_next_pred = training_dynamic_step(
            lagrangian=lagrangian,
            q_prev=q_prev_roll,
            q_curr=q_curr_roll,
            alpha=alpha,
            solver_steps=solver_steps,
            detach_inputs=detach_between_steps,
        )
        solver_terms.append(F.mse_loss(q_next_pred, q_sequence[:, step_idx + 2]))
        if autoencoder is not None and target_frames is not None:
            pred_frame = decode_latent(
                autoencoder,
                q_next_pred,
                enable_grad=enable_decode_grad,
            )
            pred_recon_terms.append(F.mse_loss(pred_frame, target_frames[:, step_idx + 2]))
        if detach_between_steps:
            q_prev_roll = q_curr_roll.detach()
            q_curr_roll = q_next_pred.detach()
        else:
            q_prev_roll = q_curr_roll
            q_curr_roll = q_next_pred
    solver_loss = torch.stack(solver_terms).mean()
    pred_recon_loss = (
        q_sequence.new_zeros(())
        if not pred_recon_terms
        else torch.stack(pred_recon_terms).mean()
    )

    return anchor_loss, solver_loss, del_loss, pred_recon_loss


def rollout_with_predictor(predictor, q_sequence):
    predictions = [q_sequence[:, 0], q_sequence[:, 1]]
    q_prev = q_sequence[:, 0]
    q_curr = q_sequence[:, 1]
    for _ in range(q_sequence.shape[1] - 2):
        q_next = predictor(q_prev, q_curr)
        predictions.append(q_next)
        q_prev, q_curr = q_curr, q_next
    return torch.stack(predictions, dim=1)


def compute_direct_predictor_sequence_losses(
    predictor,
    q_sequence,
    autoencoder=None,
    target_frames=None,
    enable_decode_grad=False,
):
    anchor_loss = F.mse_loss(2 * q_sequence[:, 1:-1] - q_sequence[:, :-2], q_sequence[:, 2:])

    teacher_terms = []
    for step_idx in range(q_sequence.shape[1] - 2):
        teacher_terms.append(
            F.mse_loss(
                predictor(q_sequence[:, step_idx], q_sequence[:, step_idx + 1]),
                q_sequence[:, step_idx + 2],
            )
        )
    teacher_loss = torch.stack(teacher_terms).mean()

    rollout = rollout_with_predictor(predictor, q_sequence)
    rollout_loss = F.mse_loss(rollout[:, 2:], q_sequence[:, 2:])

    pred_recon_terms = []
    if autoencoder is not None and target_frames is not None:
        for step_idx in range(q_sequence.shape[1] - 2):
            pred_frame = decode_latent(
                autoencoder,
                rollout[:, step_idx + 2],
                enable_grad=enable_decode_grad,
            )
            pred_recon_terms.append(F.mse_loss(pred_frame, target_frames[:, step_idx + 2]))
    pred_recon_loss = (
        q_sequence.new_zeros(())
        if not pred_recon_terms
        else torch.stack(pred_recon_terms).mean()
    )
    return anchor_loss, teacher_loss, rollout_loss, pred_recon_loss


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


def compute_latent_velocity(latent_sequence):
    """Mean absolute step-to-step displacement across the latent sequence.

    latent_sequence: (T, B, C, H, W) or (B, T, C, H, W).
    Returns a scalar. Trending toward zero means latent is collapsing to static.
    """
    with torch.no_grad():
        if latent_sequence.dim() == 5 and latent_sequence.shape[1] >= 2:
            # (B, T, C, H, W)
            diff = (latent_sequence[:, 1:] - latent_sequence[:, :-1]).abs()
        elif latent_sequence.dim() == 4 and latent_sequence.shape[0] >= 2:
            # (T, C, H, W) single example
            diff = (latent_sequence[1:] - latent_sequence[:-1]).abs()
        else:
            return None
        return diff.mean().item()


def compute_solver_residual_norm(lagrangian, q_prev, q_curr, q_next_pred):
    """DEL residual norm evaluated at the solver-predicted next state.

    Large value → solver is not finding the DEL minimum.
    Near zero with poor rollout → Lagrangian minimum is at the wrong place.
    """
    with torch.no_grad():
        q_curr_probe = q_curr.detach().requires_grad_(True)
        l_prev = lagrangian(q_prev.detach(), q_curr_probe)
        d2_prev = torch.autograd.grad(l_prev.sum(), q_curr_probe, create_graph=False)[0]
        q_curr_probe2 = q_curr.detach().requires_grad_(True)
        l_curr = lagrangian(q_curr_probe2, q_next_pred.detach())
        d1_curr = torch.autograd.grad(l_curr.sum(), q_curr_probe2, create_graph=False)[0]
        residual = d2_prev + d1_curr
        return residual.norm(dim=list(range(1, residual.dim()))).mean().item()


def apply_state_representation(state_projector, latent):
    return latent if state_projector is None else state_projector(latent)


def compute_sigreg_loss(sigreg, latent_sequence):
    if sigreg is None:
        return latent_sequence.new_zeros(())

    if latent_sequence.dim() == 4:
        projections = latent_sequence.unsqueeze(1).flatten(2)
    elif latent_sequence.dim() == 5:
        projections = latent_sequence.flatten(2)
    else:
        raise ValueError(
            "Expected latent sequence with shape (T, C, H, W) or (T, B, C, H, W), got "
            f"{tuple(latent_sequence.shape)}."
        )
    return sigreg(projections)


def module_is_trainable(module):
    return module is not None and any(param.requires_grad for param in module.parameters())


def compute_reconstruction_loss(autoencoder, latent, frames):
    recon = autoencoder.decode(latent)
    if recon.shape != frames.shape:
        min_frames = min(recon.shape[0], frames.shape[0])
        recon = recon[:min_frames]
        frames = frames[:min_frames]
    return F.mse_loss(recon, frames), recon


def decode_latent(autoencoder, latent, enable_grad=False):
    try:
        return autoencoder.decode(latent, enable_grad=enable_grad)
    except TypeError:
        return autoencoder.decode(latent)


def unwrap_dataset(dataset):
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    if hasattr(current, "base_dataset"):
        return current.base_dataset
    return current


def sample_dataset_sequence(dataset):
    sample = dataset[0]
    if torch.is_tensor(sample):
        return sample
    return None


def encode_frame_sequence_batch(autoencoder, frames, trainable):
    batch_size, time_steps = frames.shape[:2]
    flat_frames = frames.flatten(0, 1)
    if trainable:
        latent = autoencoder(flat_frames)
    else:
        with torch.no_grad():
            latent = autoencoder(flat_frames)
    return latent.view(batch_size, time_steps, *latent.shape[1:])


def apply_state_representation_sequence(state_projector, latent_sequence):
    if state_projector is None:
        return latent_sequence
    batch_size, time_steps = latent_sequence.shape[:2]
    flat = latent_sequence.flatten(0, 1)
    flat = state_projector(flat)
    return flat.view(batch_size, time_steps, *flat.shape[1:])


def evaluate_validation(
    dynamics_model,
    dynamics_mode,
    state_projector,
    autoencoder,
    val_loader,
    device,
    sequence_mode,
    autoencoder_trainable,
    lambda_del,
    lambda_solver_mse,
    lambda_recon,
    lambda_pred_recon,
    lambda_sigreg,
    sigreg,
    solver_alpha,
    training_solver_steps,
    solver_microbatch_size,
    max_batches,
):
    if val_loader is None:
        return None

    metric_sums = {
        "anchor_mse": 0.0,
        "solver_mse": 0.0,
        "recon": 0.0,
        "pred_recon": 0.0,
        "del": 0.0,
        "sigreg": 0.0,
        "total": 0.0,
    }
    num_batches = 0

    with _temporary_eval_mode(dynamics_model, state_projector, autoencoder):
        for batch_idx, batch in enumerate(val_loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break

            if sequence_mode:
                frames = batch.to(device)
                q_sequence_base = encode_frame_sequence_batch(autoencoder, frames, autoencoder_trainable)
                q_sequence = apply_state_representation_sequence(state_projector, q_sequence_base)

                if lambda_recon > 0.0:
                    loss_recon, _ = compute_reconstruction_loss(
                        autoencoder,
                        q_sequence_base.flatten(0, 1),
                        frames.flatten(0, 1),
                    )
                else:
                    loss_recon = q_sequence.new_zeros(())

                if dynamics_mode == "lagrangian":
                    loss_anchor, loss_solver, loss_del, loss_pred_recon = compute_batched_sequence_losses(
                        lagrangian=dynamics_model,
                        q_sequence=q_sequence,
                        alpha=solver_alpha,
                        solver_steps=training_solver_steps,
                        detach_between_steps=True,
                        autoencoder=autoencoder if lambda_pred_recon > 0.0 else None,
                        target_frames=frames if lambda_pred_recon > 0.0 else None,
                        enable_decode_grad=False,
                    )
                else:
                    loss_anchor, teacher_loss, rollout_loss, loss_pred_recon = compute_direct_predictor_sequence_losses(
                        predictor=dynamics_model,
                        q_sequence=q_sequence,
                        autoencoder=autoencoder if lambda_pred_recon > 0.0 else None,
                        target_frames=frames if lambda_pred_recon > 0.0 else None,
                        enable_decode_grad=False,
                    )
                    loss_solver = 0.5 * (teacher_loss + rollout_loss)
                    loss_del = q_sequence.new_zeros(())
                loss_sigreg = lambda_sigreg * compute_sigreg_loss(
                    sigreg,
                    q_sequence.transpose(0, 1),
                )
                loss_total = (
                    (0.0 if dynamics_mode == "direct_predictor" else lambda_del) * loss_del
                    + lambda_solver_mse * loss_solver
                    + lambda_recon * loss_recon
                    + lambda_pred_recon * loss_pred_recon
                    + loss_sigreg
                )

                metric_sums["anchor_mse"] += loss_anchor.item()
                metric_sums["solver_mse"] += loss_solver.item()
                metric_sums["recon"] += loss_recon.item()
                metric_sums["pred_recon"] += loss_pred_recon.item()
                metric_sums["del"] += loss_del.item()
                metric_sums["sigreg"] += loss_sigreg.item()
                metric_sums["total"] += loss_total.item()
            else:
                o_prev, o_curr, o_next_true = batch
                o_prev = o_prev.to(device)
                o_curr = o_curr.to(device)
                o_next_true = o_next_true.to(device)

                with torch.no_grad():
                    q_prev_base = autoencoder(o_prev)
                    q_curr_base = autoencoder(o_curr)
                    q_next_true_base = autoencoder(o_next_true)
                    q_prev = apply_state_representation(state_projector, q_prev_base)
                    q_curr = apply_state_representation(state_projector, q_curr_base)
                    q_next_true = apply_state_representation(state_projector, q_next_true_base)

                    if lambda_recon > 0.0:
                        loss_recon_prev, _ = compute_reconstruction_loss(autoencoder, q_prev_base, o_prev)
                        loss_recon_curr, _ = compute_reconstruction_loss(autoencoder, q_curr_base, o_curr)
                        loss_recon_next, _ = compute_reconstruction_loss(autoencoder, q_next_true_base, o_next_true)
                        batch_recon_loss = (loss_recon_prev + loss_recon_curr + loss_recon_next) / 3.0
                    else:
                        batch_recon_loss = q_prev.new_zeros(())

                batch_size = q_prev.shape[0]
                batch_anchor_sum = 0.0
                batch_solver_sum = 0.0
                batch_recon_sum = float(batch_recon_loss.item())
                batch_pred_recon_sum = 0.0
                batch_del_sum = 0.0
                batch_sigreg_sum = 0.0
                batch_total_sum = 0.0

                for start_idx in range(0, batch_size, solver_microbatch_size):
                    end_idx = min(start_idx + solver_microbatch_size, batch_size)
                    q_prev_mb = q_prev[start_idx:end_idx]
                    q_curr_mb = q_curr[start_idx:end_idx]
                    q_next_true_mb = q_next_true[start_idx:end_idx]

                    anchor_pred_mb = 2 * q_curr_mb - q_prev_mb
                    loss_anchor_mse_mb = F.mse_loss(anchor_pred_mb, q_next_true_mb)

                    if dynamics_mode == "lagrangian":
                        if lambda_solver_mse > 0.0:
                            q_next_pred = training_dynamic_step(
                                lagrangian=dynamics_model,
                                q_prev=q_prev_mb,
                                q_curr=q_curr_mb,
                                alpha=solver_alpha,
                                solver_steps=training_solver_steps,
                                detach_inputs=True,
                            )
                            loss_solver_mse_mb = F.mse_loss(q_next_pred, q_next_true_mb)
                        else:
                            q_next_pred = None
                            loss_solver_mse_mb = q_prev_mb.new_zeros(())
                    else:
                        q_next_pred = dynamics_model(q_prev_mb, q_curr_mb)
                        loss_solver_mse_mb = F.mse_loss(q_next_pred, q_next_true_mb)

                    if lambda_pred_recon > 0.0:
                        if q_next_pred is None:
                            q_next_pred = training_dynamic_step(
                                lagrangian=dynamics_model,
                                q_prev=q_prev_mb,
                                q_curr=q_curr_mb,
                                alpha=solver_alpha,
                                solver_steps=training_solver_steps,
                                detach_inputs=True,
                            )
                        pred_frame_mb = decode_latent(autoencoder, q_next_pred, enable_grad=False)
                        loss_pred_recon_mb = F.mse_loss(
                            pred_frame_mb,
                            o_next_true[start_idx:end_idx],
                        )
                    else:
                        loss_pred_recon_mb = q_prev_mb.new_zeros(())

                    if dynamics_mode == "lagrangian":
                        residual_true_mb = calculate_del_residual(
                            lagrangian=dynamics_model,
                            q_prev=q_prev_mb,
                            q_curr=q_curr_mb,
                            q_next=q_next_true_mb,
                        )
                        loss_del_mb = residual_true_mb.pow(2).mean()
                    else:
                        loss_del_mb = q_prev_mb.new_zeros(())
                    loss_sigreg_mb = lambda_sigreg * compute_sigreg_loss(
                        sigreg,
                        torch.stack([q_prev_mb, q_curr_mb, q_next_true_mb], dim=0),
                    )
                    loss_total_mb = (
                        (0.0 if dynamics_mode == "direct_predictor" else lambda_del) * loss_del_mb
                        + lambda_solver_mse * loss_solver_mse_mb
                        + lambda_pred_recon * loss_pred_recon_mb
                        + loss_sigreg_mb
                    )

                    microbatch_size = q_prev_mb.shape[0]
                    batch_anchor_sum += loss_anchor_mse_mb.item() * microbatch_size
                    batch_solver_sum += loss_solver_mse_mb.item() * microbatch_size
                    batch_del_sum += loss_del_mb.item() * microbatch_size
                    batch_pred_recon_sum += loss_pred_recon_mb.item() * microbatch_size
                    batch_sigreg_sum += loss_sigreg_mb.item() * microbatch_size
                    batch_total_sum += loss_total_mb.item() * microbatch_size

                metric_sums["anchor_mse"] += batch_anchor_sum / batch_size
                metric_sums["solver_mse"] += batch_solver_sum / batch_size
                metric_sums["recon"] += batch_recon_sum
                metric_sums["pred_recon"] += batch_pred_recon_sum / batch_size
                metric_sums["del"] += batch_del_sum / batch_size
                metric_sums["sigreg"] += batch_sigreg_sum / batch_size
                metric_sums["total"] += batch_total_sum / batch_size + lambda_recon * batch_recon_sum
            num_batches += 1

    if num_batches == 0:
        return None

    metrics = {key: value / num_batches for key, value in metric_sums.items()}
    metrics["del_to_anchor_mse"] = metrics["del"] / max(metrics["anchor_mse"], 1e-12)
    return metrics


def infer_latent_shape(encoder, state_projector, dataset, device):
    sample = dataset[0]
    if torch.is_tensor(sample):
        if sample.dim() != 4:
            raise ValueError(
                f"Expected sequence sample with shape (T, C, H, W), got {tuple(sample.shape)}."
            )
        sample_prev = sample[0]
    else:
        sample_prev, _, _ = sample
    sample_prev = sample_prev.unsqueeze(0).to(device)
    with _temporary_eval_mode(encoder, state_projector), torch.no_grad():
        latent = apply_state_representation(state_projector, encoder(sample_prev))
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
    if not getattr(autoencoder, "can_decode", True):
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
            "Autoencoder reconstruction shape mismatch: input=%s recon=%s. "
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
    return torch.cat(
        [
            frame_batch[i : i + 1].repeat(repeat_count, 1, 1, 1)
            for i in range(frame_batch.shape[0])
        ],
        dim=0,
    )


def _rollout_latents(dynamics_model, dynamics_mode, q_prev, q_curr, alpha, solver_steps, rollout_steps):
    rollout = []
    q_prev_roll = q_prev.detach()
    q_curr_roll = q_curr.detach()

    for _ in range(rollout_steps):
        if dynamics_mode == "lagrangian":
            q_next_roll = forward_dynamic_step(
                lagrangian=dynamics_model,
                q_prev=q_prev_roll,
                q_curr=q_curr_roll,
                alpha=alpha,
                solver_steps=solver_steps,
            ).detach()
        else:
            q_next_roll = dynamics_model(q_prev_roll, q_curr_roll).detach()
        rollout.append(q_next_roll)
        q_prev_roll, q_curr_roll = q_curr_roll, q_next_roll

    return rollout


def _uniform_sample_indices(length, sample_count):
    sample_count = max(1, min(int(sample_count), int(length)))
    return torch.linspace(0, length - 1, steps=sample_count).round().long()


def _default_run_name(output_dir):
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_name and slurm_job_id:
        return f"{slurm_job_name}-{slurm_job_id}"
    if slurm_job_name:
        return slurm_job_name
    return output_dir.name


def _init_wandb(cfg, output_dir):
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not bool(wandb_cfg.get("enabled", False)):
        return None

    if wandb is None:
        raise ImportError(
            "wandb logging is enabled but the `wandb` package is not installed. "
            "Install it in the training environment or set wandb.enabled=false."
        )

    run_name = wandb_cfg.get("name") or _default_run_name(output_dir)
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
    if os.environ.get("SLURM_JOB_ID"):
        run.summary["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
    if os.environ.get("SLURM_JOB_NAME"):
        run.summary["slurm_job_name"] = os.environ["SLURM_JOB_NAME"]
    return run


def _save_checkpoint(
    output_dir,
    name,
    lagrangian,
    autoencoder,
    state_projector,
    optimizer,
    scheduler,
    cfg,
    epoch,
    metrics,
):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": lagrangian.state_dict(),
            "autoencoder_state_dict": (
                None if not module_is_trainable(autoencoder) else autoencoder.state_dict()
            ),
            "state_projector_state_dict": (
                None if state_projector is None else state_projector.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_inference_and_save(
    dynamics_model,
    dynamics_mode,
    state_projector,
    autoencoder,
    debug_dataset,
    eval_dataset,
    epoch,
    cfg,
    device,
    save_dir,
    wandb_run=None,
    global_step=None,
    artifact_tag=None,
):
    """
    Saves sparse-sampled GT vs predicted frames plus a side-by-side rollout video.
    """
    if autoencoder is None:
        return
    if not getattr(autoencoder, "can_decode", True):
        log.info("Skipping inference visualization: encoder has no decoder.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_step = int(global_step) if global_step is not None else int(epoch)
    if artifact_tag is None:
        artifact_tag = f"epoch_{epoch:03d}"
        if global_step is not None:
            artifact_tag = f"{artifact_tag}_step_{int(global_step):07d}"
    caption_label = f"epoch {epoch}"
    if global_step is not None:
        caption_label = f"{caption_label}, step {int(global_step)}"

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

    gt_sequence = sample_dataset_sequence(debug_dataset)
    if gt_sequence is not None:
        gt_sequence = gt_sequence[:inference_video_max_frames]
    else:
        gt_sequence = eval_dataset.get_video_sequence(
            video_index=inference_video_index,
            max_frames=inference_video_max_frames,
        )
    if gt_sequence.shape[0] < 2:
        raise RuntimeError(
            "Need at least 2 frames in the evaluation video sequence for rollout."
        )

    gt_sequence_display = _to_display_range(gt_sequence)

    with _temporary_eval_mode(dynamics_model, state_projector, autoencoder):
        with torch.no_grad():
            seed_frames = gt_sequence[:2].to(device)
            seed_latents = apply_state_representation(
                state_projector,
                autoencoder(seed_frames),
            )
            q_prev_sim = seed_latents[0:1]
            q_curr_sim = seed_latents[1:2]

            rollout_latents = _rollout_latents(
                dynamics_model=dynamics_model,
                dynamics_mode=dynamics_mode,
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
                energy_metrics = (
                    summarize_energy_drift(dynamics_model, pred_latent_sequence)
                    if dynamics_mode == "lagrangian"
                    else None
                )
                decoded_future = autoencoder.decode(future_latents)
                pred_future_display = _to_display_range(decoded_future)
                pred_sequence_display = torch.cat(
                    [gt_sequence_display[:2], pred_future_display],
                    dim=0,
                )
            else:
                pred_latent_sequence = torch.cat([q_prev_sim, q_curr_sim], dim=0)
                energy_metrics = (
                    summarize_energy_drift(dynamics_model, pred_latent_sequence)
                    if dynamics_mode == "lagrangian"
                    else None
                )
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
        comparison_path = save_dir / f"{artifact_tag}_comparison.png"
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
            stem=f"{artifact_tag}_rollout",
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
            media_payload["media/predicted_comparison"] = wandb.Image(
                str(comparison_path),
                caption=(
                    f"{caption_label}: top row GT sparse samples, bottom row predicted sparse samples"
                ),
            )
        if video_path is not None:
            media_payload["media/predicted_rollout"] = wandb.Video(
                str(video_path),
                caption=f"{caption_label}: left GT sequence, right predicted sequence",
                fps=inference_video_fps,
                format=video_path.suffix.lstrip("."),
            )
        if media_payload:
            wandb_run.log(media_payload, step=metrics_step)
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
                step=metrics_step,
            )

    return comparison_path, video_path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    seed = int(cfg.training.get("seed", 0))
    set_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    requested_device = str(cfg.training.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "training.device is set to cuda, but CUDA is not available in this job. "
            "Refusing to silently fall back to CPU."
        )
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    if int(cfg.training.batch_size) <= 0:
        raise ValueError("training.batch_size must be > 0.")
    if int(cfg.training.epochs) <= 0:
        raise ValueError("training.epochs must be > 0.")
    if int(cfg.training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers must be >= 0.")
    if int(cfg.training.get("overfit_num_workers", 0)) < 0:
        raise ValueError("training.overfit_num_workers must be >= 0.")

    log.info("Starting DIALGA training on %s...", device)
    log.info("Using seed %d", seed)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    inference_dir = output_dir / "inference"
    log.info("Hydra run output directory: %s", output_dir)
    wandb_run = _init_wandb(cfg, output_dir)

    latent_source = str(cfg.model.get("latent_source", "wan_vae"))
    if latent_source == "wan_vae" and not os.path.isfile(cfg.model.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {cfg.model.vae_ckpt}")

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

    sequence_mode = int(cfg.training.get("sequence_window_length", 0)) > 0
    if sequence_mode:
        dataset = ClevrerSequenceWindowDataset(
            data_dir=data_dir,
            window_length=int(cfg.training.get("sequence_window_length", 6)),
            max_frames=int(cfg.dataset.get("video_num_frames", 128)),
            windows_per_video=int(cfg.training.get("sequence_windows_per_video", 4)),
            max_videos=int(cfg.training.get("sequence_max_videos", 0)),
            seed=seed,
        )
        log.info(
            "Sequence training mode enabled | window_length=%d | windows_per_video=%d | max_videos=%d",
            int(cfg.training.get("sequence_window_length", 6)),
            int(cfg.training.get("sequence_windows_per_video", 4)),
            int(cfg.training.get("sequence_max_videos", 0)),
        )
    else:
        dataset = ClevrerTripletDataset(
            data_dir=data_dir,
            video_num_frames=int(cfg.dataset.get("video_num_frames", 128)),
        )
    base_dataset = dataset
    train_subset_size = int(cfg.training.get("train_subset_size", 0))
    overfit_subset_size = int(cfg.training.get("overfit_subset_size", 0))
    overfit_video_index = int(cfg.training.get("overfit_video_index", -1))
    if sequence_mode and (overfit_video_index >= 0 or overfit_subset_size > 0):
        raise ValueError("Sequence training mode does not support overfit_* options.")
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
    elif train_subset_size > 0:
        subset_size = min(train_subset_size, len(dataset))
        subset_generator = torch.Generator().manual_seed(seed)
        subset_indices = torch.randperm(len(dataset), generator=subset_generator)[:subset_size].tolist()
        dataset = Subset(dataset, subset_indices)
        log.info(
            "Training subset mode enabled with %d sampled %s.",
            subset_size,
            "windows" if sequence_mode else "triplets",
        )

    val_loader = None
    val_fraction = float(cfg.training.get("val_fraction", 0.0))
    if overfit_video_index >= 0 or overfit_subset_size > 0:
        if val_fraction > 0.0:
            log.info("Skipping held-out validation split in overfit mode.")
    elif val_fraction > 0.0 and len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        val_size = min(val_size, len(dataset) - 1)
        train_size = len(dataset) - val_size
        split_generator = torch.Generator().manual_seed(seed)
        dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=split_generator,
        )
        val_batch_size = max(1, int(cfg.training.get("val_batch_size", effective_batch_size)))
        val_num_workers = int(cfg.training.get("val_num_workers", 0))
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=val_num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=val_num_workers > 0,
        )
        log.info(
            "Using held-out validation split | train samples: %d | val samples: %d",
            train_size,
            val_size,
        )

    eval_dataset = unwrap_dataset(dataset)
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

    if latent_source == "wan_vae":
        autoencoder = WanFrozenEncoder(vae_pth=cfg.model.vae_ckpt, device=device)
        log.warning(
            "WanFrozenEncoder is encoding individual frames with a video VAE. "
            "Use the reconstruction sanity check below to verify this latent is usable; "
            "object-centric CLEVRER coordinates remain the preferred physics state."
        )
    elif latent_source == "lewm_patch":
        autoencoder = LeWMPatchAutoencoder(
            image_size=int(cfg.model.get("input_image_size", 128)),
            patch_size=int(cfg.model.get("lewm_patch_size", 16)),
            embed_dim=int(cfg.model.get("lewm_embed_dim", 192)),
            latent_channels=int(cfg.model.get("lewm_latent_channels", 48)),
            depth=int(cfg.model.get("lewm_encoder_depth", 4)),
            num_heads=int(cfg.model.get("lewm_num_heads", 6)),
            mlp_ratio=float(cfg.model.get("lewm_mlp_ratio", 4.0)),
        ).to(device)
        log.info("Using optional LeWM-inspired patch autoencoder backend.")
    elif latent_source == "dino_vits14":
        autoencoder = FrozenDINOAutoencoder(
            model_name=str(cfg.model.get("dino_model_name", "dinov2_vits14")),
            input_size=int(cfg.model.get("dino_input_size", 126)),
            latent_channels=int(cfg.model.get("dino_latent_channels", 64)),
        ).to(device)
        log.info("Using frozen DINOv2 encoder backend.")
    elif latent_source == "dinov2":
        autoencoder = DinoV2FrozenEncoder(
            variant=str(cfg.model.get("dinov2_variant", "dinov2_vits14")),
            image_size=int(cfg.model.get("dinov2_image_size", 224)),
        ).to(device)
        log.info(
            "Using DinoV2FrozenEncoder | variant=%s | image_size=%d | embed_dim=%d | grid=%dx%d",
            cfg.model.get("dinov2_variant", "dinov2_vits14"),
            cfg.model.get("dinov2_image_size", 224),
            autoencoder.embed_dim,
            autoencoder.grid,
            autoencoder.grid,
        )
    else:
        raise ValueError(
            "Unsupported model.latent_source='{}'. Use 'wan_vae', 'lewm_patch', 'dino_vits14', or 'dinov2'.".format(
                latent_source
            )
        )

    autoencoder_trainable = module_is_trainable(autoencoder)
    representation_mode = str(cfg.model.get("representation", "wan_frozen"))
    if representation_mode in {"wan_frozen", "identity"}:
        state_projector = None
    elif representation_mode in {"wan_projected_sigreg", "projected_sigreg"}:
        base_latent_shape = infer_latent_shape(autoencoder, None, dataset, device)
        state_projector = ResidualStateProjector(
            channels=base_latent_shape[0],
            hidden_channels=int(cfg.model.get("state_projector_hidden_channels", 128)),
        ).to(device)
    else:
        raise ValueError(
            "Unsupported model.representation='{}'. Use 'identity' / 'wan_frozen' or "
            "'projected_sigreg' / 'wan_projected_sigreg'.".format(representation_mode)
        )

    lambda_recon = float(cfg.training.get("lambda_recon", 0.0))
    lambda_pred_recon = float(cfg.training.get("lambda_pred_recon", 0.0))
    if autoencoder_trainable and lambda_recon <= 0.0:
        raise ValueError(
            "model.latent_source={} is trainable and requires training.lambda_recon>0."
            .format(latent_source)
        )
    if not autoencoder_trainable and lambda_recon > 0.0:
        log.warning(
            "training.lambda_recon>0 was set, but the selected latent source is frozen. "
            "The reconstruction term will not update the encoder."
        )

    lambda_sigreg = float(cfg.training.get("lambda_sigreg", 0.0))
    if lambda_sigreg > 0.0 and state_projector is None:
        raise ValueError(
            "training.lambda_sigreg > 0 requires a trainable state representation. "
            "Set model.representation=projected_sigreg."
        )
    if state_projector is not None and lambda_sigreg <= 0.0:
        log.warning(
            "model.representation=%s is enabled but training.lambda_sigreg=0.0. "
            "The projector will only be shaped indirectly through the DEL losses.",
            representation_mode,
        )
    sigreg = None if lambda_sigreg <= 0.0 else SIGReg(
        knots=int(cfg.training.get("sigreg_knots", 17)),
        num_proj=int(cfg.training.get("sigreg_num_proj", 256)),
    ).to(device)
    log.info("Latent source: %s | state representation mode: %s", latent_source, representation_mode)
    if state_projector is not None:
        log.warning(
            "LeWM-style state projection is active. Inference videos decode the projected "
            "latents directly, so visual reconstructions should be treated as approximate."
        )

    save_inference_images = bool(cfg.training.get("save_inference_images", True))
    save_inference_video = bool(cfg.training.get("save_inference_video", True))
    save_inference_visuals = save_inference_images or save_inference_video
    save_final_inference = bool(cfg.training.get("save_final_inference", True))

    actual_latent_shape = infer_latent_shape(autoencoder, state_projector, dataset, device)
    configured_latent_shape = (
        cfg.model.latent_channels,
        cfg.model.latent_h,
        cfg.model.latent_w,
    )
    if actual_latent_shape != configured_latent_shape:
        log.warning(
            "Config latent shape %s does not match encoder output %s. "
            "Using encoder output for the selected latent dynamics model.",
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
        autoencoder=autoencoder,
        dataset=eval_dataset,
        device=device,
        video_index=max(overfit_video_index, 0),
        max_frames=reconstruction_check_frames,
    )
    if reconstruction_metrics is not None:
        log.info(
            "Single-frame autoencoder reconstruction check | MSE: %.6f | MAE: %.6f",
            reconstruction_metrics["mse"],
            reconstruction_metrics["mae"],
        )
        if reconstruction_metrics["mse"] > reconstruction_warn_mse:
            log.warning(
                "Autoencoder reconstruction MSE %.6f exceeds warning threshold %.6f. "
                "The latent dynamics model may be fighting an off-distribution encoder.",
                reconstruction_metrics["mse"],
                reconstruction_warn_mse,
            )
        if wandb_run is not None:
            wandb_run.summary["diagnostics/autoencoder_reconstruction_mse"] = (
                reconstruction_metrics["mse"]
            )
            wandb_run.summary["diagnostics/autoencoder_reconstruction_mae"] = (
                reconstruction_metrics["mae"]
            )

    dynamics_mode = str(cfg.model.get("dynamics_model", "lagrangian"))
    if dynamics_mode == "lagrangian":
        lagrangian = DiTLagrangian(
            latent_channels=actual_latent_shape[0],
            latent_h=actual_latent_shape[1],
            latent_w=actual_latent_shape[2],
            patch_size=cfg.model.patch_size,
            hidden_size=cfg.model.hidden_size,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            action_dim=cfg.model.action_dim,
        ).to(device)
        lagrangian.set_gradient_checkpointing(False)
    elif dynamics_mode == "direct_predictor":
        lagrangian = LatentNextStatePredictor(
            latent_channels=actual_latent_shape[0],
            hidden_channels=int(cfg.model.get("predictor_hidden_channels", 256)),
            num_blocks=int(cfg.model.get("predictor_num_blocks", 4)),
        ).to(device)
    else:
        raise ValueError(
            "Unsupported model.dynamics_model='{}'. Use 'lagrangian' or 'direct_predictor'.".format(
                dynamics_mode
            )
        )
    if dynamics_mode == "direct_predictor" and not sequence_mode:
        raise ValueError("model.dynamics_model=direct_predictor currently requires sequence training mode.")

    lambda_del = float(cfg.training.get("lambda_del", 1.0))
    lambda_solver_mse = float(cfg.training.get("lambda_solver_mse", 0.0))
    solver_supervision_enabled = lambda_solver_mse > 0.0
    if lambda_del <= 0.0 and lambda_solver_mse <= 0.0:
        raise ValueError("At least one of lambda_del or lambda_solver_mse must be > 0.")
    if not solver_supervision_enabled:
        log.warning(
            "training.lambda_solver_mse=0.0 disables direct predictive supervision. "
            "In this codebase, DEL-only training can converge to a trivial near-zero residual solution "
            "while rollout videos remain poor. Use training.lambda_solver_mse>0 for any run where next-frame "
            "or rollout quality matters. When this term is disabled, the logged solver_mse will stay at 0 by design."
        )
    if dynamics_mode == "direct_predictor" and lambda_del > 0.0:
        log.warning(
            "model.dynamics_model=direct_predictor ignores the DEL loss. "
            "Set training.lambda_del=0 for cleaner logs."
        )

    if dynamics_mode == "lagrangian":
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

    trainable_parameters = list(lagrangian.parameters())
    if autoencoder_trainable:
        trainable_parameters.extend(autoencoder.parameters())
    if state_projector is not None:
        trainable_parameters.extend(state_projector.parameters())
    optimizer = AdamW(
        trainable_parameters,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    grad_clip_norm = float(cfg.training.get("grad_clip_norm", 1.0))
    solver_alpha = float(cfg.training.get("solver_alpha", 0.1))
    training_solver_steps = int(cfg.training.get("training_solver_steps", 1))
    log_interval = int(cfg.training.get("log_interval", 10))
    diagnostics_every = int(cfg.training.get("diagnostics_every", log_interval))
    inference_every = int(cfg.training.get("inference_every", 1))
    inference_every_steps = int(cfg.training.get("inference_every_steps", 0))
    val_every = int(cfg.training.get("val_every", 1))
    val_every_steps = int(cfg.training.get("val_every_steps", 0))
    val_max_batches = int(cfg.training.get("val_max_batches", 0))
    checkpoint_every = int(cfg.training.get("checkpoint_every", 0))
    checkpoint_every_steps = int(cfg.training.get("checkpoint_every_steps", 0))
    save_last_checkpoint = bool(cfg.training.get("save_last_checkpoint", True))
    if lambda_sigreg < 0:
        raise ValueError("training.lambda_sigreg must be >= 0.")
    if lambda_pred_recon < 0:
        raise ValueError("training.lambda_pred_recon must be >= 0.")
    if inference_every < 0:
        raise ValueError("training.inference_every must be >= 0.")
    if inference_every_steps < 0:
        raise ValueError("training.inference_every_steps must be >= 0.")
    if val_every < 0:
        raise ValueError("training.val_every must be >= 0.")
    if val_every_steps < 0:
        raise ValueError("training.val_every_steps must be >= 0.")
    if val_max_batches < 0:
        raise ValueError("training.val_max_batches must be >= 0.")
    if checkpoint_every < 0:
        raise ValueError("training.checkpoint_every must be >= 0.")
    if checkpoint_every_steps < 0:
        raise ValueError("training.checkpoint_every_steps must be >= 0.")
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
    use_scheduler = bool(cfg.training.get("use_scheduler", True))
    warmup_start_factor = 0.1
    if not use_scheduler:
        scheduler = None
    elif warmup_epochs == 0:
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

    last_epoch_metrics = None
    try:
        last_inference_epoch = None
        last_validation_step = None
        global_step = 0
        is_interactive = sys.stdout.isatty()
        for epoch in range(1, cfg.training.epochs + 1):
            lagrangian.train()
            if autoencoder_trainable:
                autoencoder.train()
            if state_projector is not None:
                state_projector.train()
            epoch_total_loss = 0.0
            epoch_anchor_mse_loss = 0.0
            epoch_solver_mse_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_pred_recon_loss = 0.0
            epoch_del_loss = 0.0
            epoch_sigreg_loss = 0.0
            epoch_diag_sums = {}
            epoch_diag_count = 0
            num_steps = 0

            if overfit_video_index >= 0:
                optimizer.zero_grad(set_to_none=True)
                overfit_sequence = base_dataset.get_video_sequence(
                    video_index=overfit_video_index,
                    max_frames=int(cfg.dataset.get("video_num_frames", 128)),
                )
                if autoencoder_trainable:
                    overfit_sequence = overfit_sequence.to(device)
                    q_sequence_base = autoencoder(overfit_sequence)
                    loss_recon_tensor, _ = compute_reconstruction_loss(
                        autoencoder,
                        q_sequence_base,
                        overfit_sequence,
                    )
                else:
                    q_sequence_base = encode_video_sequence_in_chunks(
                        encoder=autoencoder,
                        frame_sequence=overfit_sequence,
                        device=device,
                        chunk_size=sequence_encode_batch_size,
                    )
                    loss_recon_tensor = q_sequence_base.new_zeros(())
                q_sequence = apply_state_representation(state_projector, q_sequence_base)

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

                loss_sigreg_tensor = lambda_sigreg * compute_sigreg_loss(sigreg, q_sequence)

                total_loss = (
                    lambda_del * loss_DEL_tensor
                    + lambda_solver_mse * loss_solver_mse_tensor
                    + lambda_recon * loss_recon_tensor
                    + loss_sigreg_tensor
                )
                total_loss.backward()

                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, max_norm=grad_clip_norm
                    )
                optimizer.step()

                epoch_anchor_mse_loss = loss_anchor_mse_tensor.item()
                epoch_solver_mse_loss = loss_solver_mse_tensor.item()
                epoch_recon_loss = loss_recon_tensor.item()
                epoch_del_loss = loss_DEL_tensor.item()
                epoch_sigreg_loss = loss_sigreg_tensor.item()
                epoch_total_loss = total_loss.item()
                component_stats = summarize_lagrangian_components(
                    lagrangian,
                    q_sequence[0:1],
                    q_sequence[1:2],
                )
                epoch_diag_sums.update(component_stats)
                epoch_diag_count = 1
                num_steps = 1
                global_step += 1
                log.info(
                    "Epoch %d overfit | anchor MSE: %.6f | solver MSE: %.6f | recon: %.6f | DEL: %.6f | SIGReg: %.6f | LR: %.6e",
                    epoch,
                    epoch_anchor_mse_loss,
                    epoch_solver_mse_loss,
                    epoch_recon_loss,
                    epoch_del_loss,
                    epoch_sigreg_loss,
                    optimizer.param_groups[0]["lr"],
                )
            else:
                pbar = tqdm(
                    loader,
                    desc=f"Epoch {epoch}/{cfg.training.epochs}",
                    disable=not is_interactive,
                    dynamic_ncols=is_interactive,
                )

                for batch_idx, batch in enumerate(pbar, start=1):
                    optimizer.zero_grad(set_to_none=True)

                    if sequence_mode:
                        frames = batch.to(device)
                        batch_size = frames.shape[0]
                        q_sequence_base = encode_frame_sequence_batch(
                            autoencoder,
                            frames,
                            autoencoder_trainable,
                        )
                        q_sequence = apply_state_representation_sequence(
                            state_projector,
                            q_sequence_base,
                        )

                        if lambda_recon > 0.0:
                            loss_recon_tensor, _ = compute_reconstruction_loss(
                                autoencoder,
                                q_sequence_base.flatten(0, 1),
                                frames.flatten(0, 1),
                            )
                        else:
                            loss_recon_tensor = q_sequence.new_zeros(())

                        if dynamics_mode == "lagrangian":
                            loss_anchor_tensor, loss_solver_tensor, loss_del_tensor, loss_pred_recon_tensor = compute_batched_sequence_losses(
                                lagrangian=lagrangian,
                                q_sequence=q_sequence,
                                alpha=solver_alpha,
                                solver_steps=training_solver_steps,
                                detach_between_steps=autoregressive_detach_between_steps,
                                autoencoder=autoencoder if lambda_pred_recon > 0.0 else None,
                                target_frames=frames if lambda_pred_recon > 0.0 else None,
                                enable_decode_grad=True,
                            )
                        else:
                            (
                                loss_anchor_tensor,
                                teacher_loss_tensor,
                                rollout_loss_tensor,
                                loss_pred_recon_tensor,
                            ) = compute_direct_predictor_sequence_losses(
                                predictor=lagrangian,
                                q_sequence=q_sequence,
                                autoencoder=autoencoder if lambda_pred_recon > 0.0 else None,
                                target_frames=frames if lambda_pred_recon > 0.0 else None,
                                enable_decode_grad=True,
                            )
                            loss_solver_tensor = 0.5 * (teacher_loss_tensor + rollout_loss_tensor)
                            loss_del_tensor = q_sequence.new_zeros(())
                        loss_sigreg_tensor = lambda_sigreg * compute_sigreg_loss(
                            sigreg,
                            q_sequence.transpose(0, 1),
                        )
                        loss_total_tensor = (
                            (0.0 if dynamics_mode == "direct_predictor" else lambda_del) * loss_del_tensor
                            + lambda_solver_mse * loss_solver_tensor
                            + lambda_recon * loss_recon_tensor
                            + lambda_pred_recon * loss_pred_recon_tensor
                            + loss_sigreg_tensor
                        )
                        loss_total_tensor.backward()

                        if diagnostics_every > 0 and batch_idx % diagnostics_every == 0:
                            lat_vel = compute_latent_velocity(q_sequence)
                            if lat_vel is not None:
                                epoch_diag_sums["latent_velocity"] = (
                                    epoch_diag_sums.get("latent_velocity", 0.0) + lat_vel
                                )

                            if dynamics_mode == "lagrangian":
                                component_stats = summarize_lagrangian_components(
                                    lagrangian,
                                    q_sequence[:, 0],
                                    q_sequence[:, 1],
                                )
                                for key, value in component_stats.items():
                                    epoch_diag_sums[key] = (
                                        epoch_diag_sums.get(key, 0.0) + value
                                    )
                                if lambda_solver_mse > 0.0:
                                    with torch.no_grad():
                                        q_next_diag = training_dynamic_step(
                                            lagrangian=lagrangian,
                                            q_prev=q_sequence[:, 0].detach(),
                                            q_curr=q_sequence[:, 1].detach(),
                                            alpha=solver_alpha,
                                            solver_steps=training_solver_steps,
                                            detach_inputs=True,
                                        ).detach()
                                    solver_resid = compute_solver_residual_norm(
                                        lagrangian,
                                        q_sequence[:, 0],
                                        q_sequence[:, 1],
                                        q_next_diag,
                                    )
                                    epoch_diag_sums["solver_residual_norm"] = (
                                        epoch_diag_sums.get("solver_residual_norm", 0.0) + solver_resid
                                    )
                            epoch_diag_count += 1

                        loss_anchor_mse = loss_anchor_tensor.item()
                        loss_solver_mse = loss_solver_tensor.item()
                        loss_recon = loss_recon_tensor.item()
                        loss_pred_recon = loss_pred_recon_tensor.item()
                        loss_DEL = loss_del_tensor.item()
                        loss_sigreg = loss_sigreg_tensor.item()
                        loss_total = loss_total_tensor.item()
                    else:
                        o_prev, o_curr, o_next_true = batch
                        o_prev = o_prev.to(device)
                        o_curr = o_curr.to(device)
                        o_next_true = o_next_true.to(device)

                        if autoencoder_trainable:
                            q_prev = None
                            q_curr = None
                            q_next_true = None
                        else:
                            with torch.no_grad():
                                q_prev = autoencoder(o_prev)
                                q_curr = autoencoder(o_curr)
                                q_next_true = autoencoder(o_next_true)

                        batch_size = o_prev.shape[0]
                        batch_anchor_mse_sum = 0.0
                        batch_solver_mse_sum = 0.0
                        batch_recon_sum = 0.0
                        batch_pred_recon_sum = 0.0
                        batch_del_sum = 0.0
                        batch_sigreg_sum = 0.0
                        batch_total_sum = 0.0

                        for start_idx in range(0, batch_size, solver_microbatch_size):
                            end_idx = min(start_idx + solver_microbatch_size, batch_size)
                            if autoencoder_trainable:
                                o_prev_mb = o_prev[start_idx:end_idx]
                                o_curr_mb = o_curr[start_idx:end_idx]
                                o_next_true_frame_mb = o_next_true[start_idx:end_idx]

                                q_prev_base_mb = autoencoder(o_prev_mb)
                                q_curr_base_mb = autoencoder(o_curr_mb)
                                q_next_true_base_mb = autoencoder(o_next_true_frame_mb)

                                loss_recon_prev_mb, _ = compute_reconstruction_loss(
                                    autoencoder,
                                    q_prev_base_mb,
                                    o_prev_mb,
                                )
                                loss_recon_curr_mb, _ = compute_reconstruction_loss(
                                    autoencoder,
                                    q_curr_base_mb,
                                    o_curr_mb,
                                )
                                loss_recon_next_mb, _ = compute_reconstruction_loss(
                                    autoencoder,
                                    q_next_true_base_mb,
                                    o_next_true_frame_mb,
                                )
                                loss_recon_mb = (
                                    loss_recon_prev_mb + loss_recon_curr_mb + loss_recon_next_mb
                                ) / 3.0

                                q_prev_mb = apply_state_representation(state_projector, q_prev_base_mb)
                                q_curr_mb = apply_state_representation(state_projector, q_curr_base_mb)
                                q_next_true_mb = apply_state_representation(state_projector, q_next_true_base_mb)
                            else:
                                loss_recon_mb = o_prev.new_zeros(())
                                q_prev_mb = apply_state_representation(
                                    state_projector,
                                    q_prev[start_idx:end_idx],
                                )
                                q_curr_mb = apply_state_representation(
                                    state_projector,
                                    q_curr[start_idx:end_idx],
                                )
                                q_next_true_mb = apply_state_representation(
                                    state_projector,
                                    q_next_true[start_idx:end_idx],
                                )

                            anchor_pred_mb = 2 * q_curr_mb - q_prev_mb
                            loss_anchor_mse_mb = F.mse_loss(
                                anchor_pred_mb,
                                q_next_true_mb,
                            )

                            if dynamics_mode == "lagrangian":
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
                                    q_next_pred = None
                                    loss_solver_mse_mb = q_prev_mb.new_zeros(())

                                if lambda_pred_recon > 0.0:
                                    if q_next_pred is None:
                                        q_next_pred = training_dynamic_step(
                                            lagrangian=lagrangian,
                                            q_prev=q_prev_mb,
                                            q_curr=q_curr_mb,
                                            alpha=solver_alpha,
                                            solver_steps=training_solver_steps,
                                            detach_inputs=True,
                                        )
                                    loss_pred_recon_mb = F.mse_loss(
                                        decode_latent(autoencoder, q_next_pred, enable_grad=True),
                                        o_next_true[start_idx:end_idx],
                                    )
                                else:
                                    loss_pred_recon_mb = q_prev_mb.new_zeros(())

                                residual_true_mb = calculate_del_residual(
                                    lagrangian=lagrangian,
                                    q_prev=q_prev_mb,
                                    q_curr=q_curr_mb,
                                    q_next=q_next_true_mb,
                                )
                                loss_DEL_mb = residual_true_mb.pow(2).mean()
                            else:
                                q_next_pred = lagrangian(q_prev_mb, q_curr_mb)
                                loss_solver_mse_mb = F.mse_loss(q_next_pred, q_next_true_mb)
                                loss_pred_recon_mb = (
                                    q_prev_mb.new_zeros(())
                                    if lambda_pred_recon <= 0.0
                                    else F.mse_loss(
                                        decode_latent(autoencoder, q_next_pred, enable_grad=True),
                                        o_next_true[start_idx:end_idx],
                                    )
                                )
                                loss_DEL_mb = q_prev_mb.new_zeros(())
                            loss_sigreg_mb = lambda_sigreg * compute_sigreg_loss(
                                sigreg,
                                torch.stack([q_prev_mb, q_curr_mb, q_next_true_mb], dim=0),
                            )
                            total_loss_mb = (
                                (0.0 if dynamics_mode == "direct_predictor" else lambda_del) * loss_DEL_mb
                                + lambda_solver_mse * loss_solver_mse_mb
                                + lambda_recon * loss_recon_mb
                                + lambda_pred_recon * loss_pred_recon_mb
                                + loss_sigreg_mb
                            )
                            scaled_loss = total_loss_mb * (q_prev_mb.shape[0] / batch_size)
                            scaled_loss.backward()
                            batch_anchor_mse_sum += (
                                loss_anchor_mse_mb.item() * q_prev_mb.shape[0]
                            )
                            batch_solver_mse_sum += (
                                loss_solver_mse_mb.item() * q_prev_mb.shape[0]
                            )
                            batch_recon_sum += loss_recon_mb.item() * q_prev_mb.shape[0]
                            batch_pred_recon_sum += loss_pred_recon_mb.item() * q_prev_mb.shape[0]
                            batch_del_sum += loss_DEL_mb.item() * q_prev_mb.shape[0]
                            batch_sigreg_sum += loss_sigreg_mb.item() * q_prev_mb.shape[0]
                            batch_total_sum += total_loss_mb.item() * q_prev_mb.shape[0]

                            if (
                                dynamics_mode == "lagrangian"
                                and diagnostics_every > 0
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

                        loss_anchor_mse = batch_anchor_mse_sum / batch_size
                        loss_solver_mse = batch_solver_mse_sum / batch_size
                        loss_recon = batch_recon_sum / batch_size
                        loss_pred_recon = batch_pred_recon_sum / batch_size
                        loss_DEL = batch_del_sum / batch_size
                        loss_sigreg = batch_sigreg_sum / batch_size
                        loss_total = batch_total_sum / batch_size

                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters, max_norm=grad_clip_norm
                        )
                    optimizer.step()

                    epoch_anchor_mse_loss += loss_anchor_mse
                    epoch_solver_mse_loss += loss_solver_mse
                    epoch_recon_loss += loss_recon
                    epoch_pred_recon_loss += loss_pred_recon
                    epoch_del_loss += loss_DEL
                    epoch_sigreg_loss += loss_sigreg
                    epoch_total_loss += loss_total
                    num_steps += 1
                    global_step += 1

                    postfix = {
                        "Total": f"{loss_total:.4f}",
                        "AnchorMSE": f"{loss_anchor_mse:.4f}",
                        "Recon": f"{loss_recon:.4f}",
                        "PredRec": f"{loss_pred_recon:.4f}",
                        "DEL": f"{loss_DEL:.4f}",
                        "SIGReg": f"{loss_sigreg:.4f}",
                        "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                    if lambda_solver_mse > 0.0:
                        postfix["SolverMSE"] = f"{loss_solver_mse:.4f}"
                    if epoch_diag_count > 0 and "mass_mean" in epoch_diag_sums:
                        postfix["Mass"] = (
                            f"{epoch_diag_sums['mass_mean'] / epoch_diag_count:.3f}"
                        )
                    pbar.set_postfix(postfix)

                    if not is_interactive and log_interval > 0 and batch_idx % log_interval == 0:
                        log.info(
                            "Epoch %d step %d | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | recon: %.6f | pred_recon: %.6f | DEL: %.6f | SIGReg: %.6f | LR: %.6e",
                            epoch,
                            batch_idx,
                            loss_total,
                            loss_anchor_mse,
                            loss_solver_mse,
                            loss_recon,
                            loss_pred_recon,
                            loss_DEL,
                            loss_sigreg,
                            optimizer.param_groups[0]["lr"],
                        )

                    if wandb_run is not None and log_interval > 0 and global_step % log_interval == 0:
                        running_anchor_mse = epoch_anchor_mse_loss / num_steps
                        running_solver_mse = epoch_solver_mse_loss / num_steps
                        running_recon = epoch_recon_loss / num_steps
                        running_pred_recon = epoch_pred_recon_loss / num_steps
                        running_del = epoch_del_loss / num_steps
                        running_sigreg = epoch_sigreg_loss / num_steps
                        wandb_payload = {
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train_step/total": loss_total,
                            "train_step/anchor_mse": loss_anchor_mse,
                            "train_step/solver_mse": loss_solver_mse,
                            "train_step/recon": loss_recon,
                            "train_step/pred_recon": loss_pred_recon,
                            "train_step/del": loss_DEL,
                            "train_step/sigreg": loss_sigreg,
                            "train_step/lr": optimizer.param_groups[0]["lr"],
                            "train_running/total": epoch_total_loss / num_steps,
                            "train_running/anchor_mse": running_anchor_mse,
                            "train_running/solver_mse": running_solver_mse,
                            "train_running/recon": running_recon,
                            "train_running/pred_recon": running_pred_recon,
                            "train_running/del": running_del,
                            "train_running/sigreg": running_sigreg,
                            "train_running/del_to_anchor_mse": (
                                running_del / max(running_anchor_mse, 1e-12)
                            ),
                        }
                        if epoch_diag_count > 0:
                            wandb_payload.update(
                                {
                                    f"diagnostics/{key}": value / epoch_diag_count
                                    for key, value in epoch_diag_sums.items()
                                }
                        )
                        wandb_run.log(wandb_payload, step=global_step)

                    if (
                        val_loader is not None
                        and val_every_steps > 0
                        and global_step % val_every_steps == 0
                        and last_validation_step != global_step
                    ):
                        val_metrics = evaluate_validation(
                            dynamics_model=lagrangian,
                            dynamics_mode=dynamics_mode,
                            state_projector=state_projector,
                            autoencoder=autoencoder,
                            val_loader=val_loader,
                            device=device,
                            sequence_mode=sequence_mode,
                            autoencoder_trainable=autoencoder_trainable,
                            lambda_del=lambda_del,
                            lambda_solver_mse=lambda_solver_mse,
                            lambda_recon=lambda_recon,
                            lambda_pred_recon=lambda_pred_recon,
                            lambda_sigreg=lambda_sigreg,
                            sigreg=sigreg,
                            solver_alpha=solver_alpha,
                            training_solver_steps=training_solver_steps,
                            solver_microbatch_size=solver_microbatch_size,
                            max_batches=val_max_batches,
                        )
                        if val_metrics is not None:
                            log.info(
                                "Val step %d | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | recon: %.6f | pred_recon: %.6f | DEL: %.6f | SIGReg: %.6f",
                                global_step,
                                val_metrics["total"],
                                val_metrics["anchor_mse"],
                                val_metrics["solver_mse"],
                                val_metrics["recon"],
                                val_metrics["pred_recon"],
                                val_metrics["del"],
                                val_metrics["sigreg"],
                            )
                            if wandb_run is not None:
                                wandb_run.log(
                                    {
                                        "val/total": val_metrics["total"],
                                        "val/anchor_mse": val_metrics["anchor_mse"],
                                        "val/solver_mse": val_metrics["solver_mse"],
                                        "val/recon": val_metrics["recon"],
                                        "val/pred_recon": val_metrics["pred_recon"],
                                        "val/del": val_metrics["del"],
                                        "val/sigreg": val_metrics["sigreg"],
                                        "val/del_to_anchor_mse": val_metrics["del_to_anchor_mse"],
                                        "val/global_step": global_step,
                                        "val/epoch": epoch,
                                    },
                                    step=global_step,
                                )
                            last_validation_step = global_step

                    if (
                        save_inference_visuals
                        and inference_every_steps > 0
                        and global_step % inference_every_steps == 0
                    ):
                        run_inference_and_save(
                            dynamics_model=lagrangian,
                            dynamics_mode=dynamics_mode,
                            state_projector=state_projector,
                            autoencoder=autoencoder,
                            debug_dataset=dataset,
                            eval_dataset=eval_dataset,
                            epoch=epoch,
                            cfg=cfg,
                            device=device,
                            save_dir=inference_dir,
                            wandb_run=wandb_run,
                            global_step=global_step,
                            artifact_tag=f"step_{global_step:07d}",
                        )
                        last_inference_epoch = epoch

                    if (
                        checkpoint_every_steps > 0
                        and global_step % checkpoint_every_steps == 0
                    ):
                        step_metrics = {
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train_step/anchor_mse": loss_anchor_mse,
                            "train_step/solver_mse": loss_solver_mse,
                            "train_step/recon": loss_recon,
                            "train_step/pred_recon": loss_pred_recon,
                            "train_step/del": loss_DEL,
                            "train_step/sigreg": loss_sigreg,
                            "train_step/lr": optimizer.param_groups[0]["lr"],
                        }
                        checkpoint_path = _save_checkpoint(
                            output_dir=output_dir,
                            name=f"step_{global_step:07d}.pt",
                            lagrangian=lagrangian,
                            autoencoder=autoencoder,
                            state_projector=state_projector,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            cfg=cfg,
                            epoch=epoch,
                            metrics=step_metrics,
                        )
                        log.info("Saved step checkpoint to %s", checkpoint_path)

            if num_steps == 0:
                raise RuntimeError(
                    "The dataloader produced zero batches. Check that the dataset contains "
                    "enough frame triplets for the configured batch size."
                )

            epoch_total_avg = epoch_total_loss / num_steps
            epoch_anchor_mse_avg = epoch_anchor_mse_loss / num_steps
            epoch_solver_mse_avg = epoch_solver_mse_loss / num_steps
            epoch_recon_avg = epoch_recon_loss / num_steps
            epoch_pred_recon_avg = epoch_pred_recon_loss / num_steps
            epoch_del_avg = epoch_del_loss / num_steps
            epoch_sigreg_avg = epoch_sigreg_loss / num_steps
            del_to_anchor_ratio = epoch_del_avg / max(epoch_anchor_mse_avg, 1e-12)
            diag_avgs = (
                {
                    key: value / epoch_diag_count
                    for key, value in epoch_diag_sums.items()
                }
                if epoch_diag_count > 0
                else {}
            )
            last_epoch_metrics = {
                "train/total": epoch_total_avg,
                "train/anchor_mse": epoch_anchor_mse_avg,
                "train/solver_mse": epoch_solver_mse_avg,
                "train/recon": epoch_recon_avg,
                "train/pred_recon": epoch_pred_recon_avg,
                "train/del": epoch_del_avg,
                "train/sigreg": epoch_sigreg_avg,
                "train/del_to_anchor_mse": del_to_anchor_ratio,
                "train/lr": optimizer.param_groups[0]["lr"],
                **{f"diagnostics/{key}": value for key, value in diag_avgs.items()},
            }

            log.info(
                "Epoch %d complete | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | recon: %.6f | pred_recon: %.6f | "
                "DEL: %.6f | SIGReg: %.6f | DEL/anchor: %.6e | LR: %.6e",
                epoch,
                epoch_total_avg,
                epoch_anchor_mse_avg,
                epoch_solver_mse_avg,
                epoch_recon_avg,
                epoch_pred_recon_avg,
                epoch_del_avg,
                epoch_sigreg_avg,
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
                    "train/global_step": global_step,
                    **last_epoch_metrics,
                }
                wandb_run.log(wandb_payload, step=global_step)

            if (
                val_loader is not None
                and val_every > 0
                and epoch % val_every == 0
                and last_validation_step != global_step
            ):
                val_metrics = evaluate_validation(
                    dynamics_model=lagrangian,
                    dynamics_mode=dynamics_mode,
                    state_projector=state_projector,
                    autoencoder=autoencoder,
                    val_loader=val_loader,
                    device=device,
                    sequence_mode=sequence_mode,
                    autoencoder_trainable=autoencoder_trainable,
                    lambda_del=lambda_del,
                    lambda_solver_mse=lambda_solver_mse,
                    lambda_recon=lambda_recon,
                    lambda_pred_recon=lambda_pred_recon,
                    lambda_sigreg=lambda_sigreg,
                    sigreg=sigreg,
                    solver_alpha=solver_alpha,
                    training_solver_steps=training_solver_steps,
                    solver_microbatch_size=solver_microbatch_size,
                    max_batches=val_max_batches,
                )
                if val_metrics is not None:
                    log.info(
                        "Epoch %d val | total: %.6f | anchor MSE: %.6f | solver MSE: %.6f | recon: %.6f | pred_recon: %.6f | DEL: %.6f | SIGReg: %.6f",
                        epoch,
                        val_metrics["total"],
                        val_metrics["anchor_mse"],
                        val_metrics["solver_mse"],
                        val_metrics["recon"],
                        val_metrics["pred_recon"],
                        val_metrics["del"],
                        val_metrics["sigreg"],
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/total": val_metrics["total"],
                                "val/anchor_mse": val_metrics["anchor_mse"],
                                "val/solver_mse": val_metrics["solver_mse"],
                                "val/recon": val_metrics["recon"],
                                "val/pred_recon": val_metrics["pred_recon"],
                                "val/del": val_metrics["del"],
                                "val/sigreg": val_metrics["sigreg"],
                                "val/del_to_anchor_mse": val_metrics["del_to_anchor_mse"],
                                "val/global_step": global_step,
                                "val/epoch": epoch,
                            },
                            step=global_step,
                        )
                    last_validation_step = global_step

            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                checkpoint_path = _save_checkpoint(
                    output_dir=output_dir,
                    name=f"epoch_{epoch:03d}.pt",
                    lagrangian=lagrangian,
                    autoencoder=autoencoder,
                    state_projector=state_projector,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    epoch=epoch,
                    metrics=last_epoch_metrics,
                )
                log.info("Saved checkpoint to %s", checkpoint_path)

            if scheduler is not None:
                scheduler.step()

            if (
                save_inference_visuals
                and inference_every > 0
                and epoch % inference_every == 0
            ):
                run_inference_and_save(
                    dynamics_model=lagrangian,
                    dynamics_mode=dynamics_mode,
                    state_projector=state_projector,
                    autoencoder=autoencoder,
                    debug_dataset=dataset,
                    eval_dataset=eval_dataset,
                    epoch=epoch,
                    cfg=cfg,
                    device=device,
                    save_dir=inference_dir,
                    wandb_run=wandb_run,
                    global_step=global_step,
                )
                last_inference_epoch = epoch

        if (
            save_inference_visuals
            and save_final_inference
            and last_inference_epoch != cfg.training.epochs
        ):
            log.info(
                "Saving final inference artifacts to %s after training.",
                inference_dir,
            )
            run_inference_and_save(
                dynamics_model=lagrangian,
                dynamics_mode=dynamics_mode,
                state_projector=state_projector,
                autoencoder=autoencoder,
                debug_dataset=dataset,
                eval_dataset=eval_dataset,
                epoch=cfg.training.epochs,
                cfg=cfg,
                device=device,
                save_dir=inference_dir,
                wandb_run=wandb_run,
                global_step=global_step,
            )

        if save_last_checkpoint:
            checkpoint_path = _save_checkpoint(
                output_dir=output_dir,
                name="last.pt",
                lagrangian=lagrangian,
                autoencoder=autoencoder,
                state_projector=state_projector,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=cfg.training.epochs,
                metrics=last_epoch_metrics,
            )
            log.info("Saved final checkpoint to %s", checkpoint_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
