import logging
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.clevrer_states import ClevrerStateDataset
from src.data.collate import collate_trajectory_batch
from src.dynamics import ObjectLagrangian, del_residual_metric, relative_energy_drift, rollout_trajectory
from src.utils.logging import configure_logging, log_metrics


log = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def build_dataset(cfg, split):
    return ClevrerStateDataset(
        annotation_dir=cfg.dataset.annotation_dir,
        split=split,
        traj_len=cfg.dataset.traj_len,
        stride=cfg.dataset.stride,
        frames_per_video=cfg.dataset.frames_per_video,
        max_objects=cfg.dataset.max_objects,
        coordinate_mode=cfg.dataset.coordinate_mode,
        use_inside_camera_view_mask=cfg.dataset.use_inside_camera_view_mask,
        video_dir=cfg.dataset.video_dir,
    )


def build_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_trajectory_batch,
    )


def compute_batch_metrics(model, batch, cfg):
    positions = batch["positions"]
    velocities = batch["velocities"]
    attrs = batch["object_attrs"]
    slot_mask = batch["slot_mask"]

    rollout_steps = min(int(cfg.dynamics.rollout_steps), positions.shape[1] - 1)
    rollout = rollout_trajectory(
        lagrangian=model,
        q0=positions[:, 0],
        qd0=velocities[:, 0],
        attrs=attrs,
        mask=slot_mask,
        dt=cfg.dynamics.dt,
        n_steps=rollout_steps,
        return_energies=True,
    )

    pred_positions = rollout["positions"][:, 1 : rollout_steps + 1]
    target_positions = positions[:, 1 : rollout_steps + 1]
    expanded_mask = slot_mask.unsqueeze(1).expand(-1, rollout_steps, -1)
    sq_error = (pred_positions - target_positions).pow(2).sum(dim=-1)
    loss = (sq_error * expanded_mask.to(sq_error.dtype)).sum() / expanded_mask.sum().clamp_min(1)

    metrics = {
        "loss": loss,
        "energy_drift": relative_energy_drift(rollout["energies"]),
        "del_residual": del_residual_metric(
            lagrangian=model,
            positions=rollout["positions"],
            attrs=attrs,
            mask=slot_mask,
            dt=cfg.dynamics.dt,
        ),
    }

    mass = model.mass(attrs)
    valid_mass = mass[slot_mask]
    metrics["mass_mean"] = valid_mass.mean()
    metrics["mass_std"] = valid_mass.std(unbiased=False)
    metrics["mass_range"] = valid_mass.max() - valid_mass.min()
    return loss, metrics


def reduce_epoch_metrics(metric_list):
    reduced = {}
    for metrics in metric_list:
        for key, value in metrics.items():
            reduced.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in reduced.items()}


def save_checkpoint(model, optimizer, epoch, cfg, metrics):
    checkpoint_dir = Path(cfg.experiment.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_epoch(model, loader, optimizer, device, cfg, train):
    model.train(train)
    epoch_metrics = []

    if cfg.training.detect_anomaly and train:
        torch.autograd.set_detect_anomaly(True)

    batch_limit = cfg.training.limit_train_batches if train else cfg.training.limit_val_batches
    for batch_idx, batch in enumerate(loader, start=1):
        if batch_limit and batch_idx > batch_limit:
            break

        batch = move_batch_to_device(batch, device)
        loss, metrics = compute_batch_metrics(model, batch, cfg)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.grad_clip_norm > 0:
                clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip_norm)
            optimizer.step()

        detached_metrics = {key: value.detach().item() for key, value in metrics.items()}
        epoch_metrics.append(detached_metrics)

        if batch_idx % int(cfg.training.print_every) == 0:
            prefix = f"train step {batch_idx}" if train else f"val step {batch_idx}"
            log_metrics(log, prefix, detached_metrics)

    if not epoch_metrics:
        raise RuntimeError("No batches were processed. Check annotation paths and loader settings.")
    return reduce_epoch_metrics(epoch_metrics)


@hydra.main(version_base=None, config_path="../conf", config_name="object_dynamics")
def main(cfg: DictConfig):
    configure_logging()
    set_seed(int(cfg.experiment.seed))

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log.info("Starting object-state dynamics training on %s", device)
    log.info("Hydra output directory: %s", Path.cwd())

    train_dataset = build_dataset(cfg, cfg.dataset.split)
    train_loader = build_loader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        shuffle=True,
    )

    val_loader = None
    val_split = cfg.dataset.get("val_split")
    if cfg.evaluation.enabled and val_split:
        try:
            val_dataset = build_dataset(cfg, val_split)
            val_loader = build_loader(
                val_dataset,
                batch_size=int(cfg.evaluation.batch_size),
                num_workers=int(cfg.evaluation.num_workers),
                shuffle=False,
            )
        except FileNotFoundError:
            log.warning("Validation annotations for split '%s' were not found. Skipping eval.", val_split)

    model = ObjectLagrangian(
        attr_dim=train_dataset.attr_dim,
        hidden_size=int(cfg.dynamics.hidden_size),
        mass_floor=float(cfg.dynamics.mass_floor),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            train=True,
        )
        log_metrics(log, f"epoch {epoch} train", train_metrics)

        if val_loader is not None and epoch % int(cfg.training.eval_every) == 0:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                train=False,
            )
            log_metrics(log, f"epoch {epoch} val", val_metrics)
        else:
            val_metrics = None

        if epoch % int(cfg.training.save_every) == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            log.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
