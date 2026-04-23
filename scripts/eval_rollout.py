import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch

from src.data.clevrer_states import ClevrerStateDataset
from src.data.collate import collate_trajectory_batch
from src.dynamics import ObjectLagrangian, del_residual_metric, relative_energy_drift, rollout_trajectory
from src.utils.logging import configure_logging, log_metrics


log = logging.getLogger(__name__)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@hydra.main(version_base=None, config_path="../conf", config_name="object_dynamics")
def main(cfg: DictConfig):
    configure_logging()
    if not cfg.evaluation.checkpoint_path:
        raise ValueError("Set evaluation.checkpoint_path=/path/to/checkpoint.pt before running eval_rollout.py.")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    dataset = ClevrerStateDataset(
        annotation_dir=cfg.dataset.annotation_dir,
        split=cfg.evaluation.split,
        traj_len=cfg.dataset.traj_len,
        stride=cfg.dataset.stride,
        frames_per_video=cfg.dataset.frames_per_video,
        max_objects=cfg.dataset.max_objects,
        coordinate_mode=cfg.dataset.coordinate_mode,
        use_inside_camera_view_mask=cfg.dataset.use_inside_camera_view_mask,
        video_dir=cfg.dataset.video_dir,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.evaluation.batch_size),
        shuffle=False,
        num_workers=int(cfg.evaluation.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg.evaluation.num_workers) > 0,
        collate_fn=collate_trajectory_batch,
    )

    model = ObjectLagrangian(
        attr_dim=dataset.attr_dim,
        hidden_size=int(cfg.dynamics.hidden_size),
        mass_floor=float(cfg.dynamics.mass_floor),
    ).to(device)

    checkpoint = torch.load(cfg.evaluation.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = []
    max_batches = int(cfg.evaluation.max_batches)
    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches and batch_idx > max_batches:
            break

        batch = move_batch_to_device(batch, device)
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
        expanded_mask = slot_mask.unsqueeze(1).expand(-1, rollout_steps, -1).to(pred_positions.dtype)
        loss = ((pred_positions - target_positions).pow(2).sum(dim=-1) * expanded_mask).sum() / expanded_mask.sum().clamp_min(1)

        metrics.append(
            {
                "loss": loss.item(),
                "energy_drift": relative_energy_drift(rollout["energies"]).item(),
                "del_residual": del_residual_metric(
                    lagrangian=model,
                    positions=rollout["positions"],
                    attrs=attrs,
                    mask=slot_mask,
                    dt=cfg.dynamics.dt,
                ).item(),
            }
        )

    if not metrics:
        raise RuntimeError("No evaluation batches were processed.")

    reduced = {
        key: sum(batch_metrics[key] for batch_metrics in metrics) / len(metrics)
        for key in metrics[0]
    }
    log.info("Loaded checkpoint from %s", Path(cfg.evaluation.checkpoint_path))
    log_metrics(log, f"eval {cfg.evaluation.split}", reduced)


if __name__ == "__main__":
    main()
