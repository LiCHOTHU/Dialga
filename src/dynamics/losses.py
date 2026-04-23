import torch

from .rollout import rollout_trajectory


def _expand_mask(mask, num_steps):
    if mask.dim() == 2:
        return mask.unsqueeze(1).expand(-1, num_steps, -1)
    if mask.dim() == 3:
        if mask.shape[1] != num_steps:
            raise ValueError(
                f"Expected mask trajectory with {num_steps} steps, got {mask.shape[1]}."
            )
        return mask
    raise ValueError(f"Expected mask with shape (B, N) or (B, T, N), got {tuple(mask.shape)}.")


def masked_position_mse(pred, target, mask):
    mask_traj = _expand_mask(mask, pred.shape[1]).to(pred.dtype)
    sq_error = (pred - target).pow(2).sum(dim=-1)
    denom = mask_traj.sum().clamp_min(1.0)
    return (sq_error * mask_traj).sum() / denom


def relative_energy_drift(energies):
    initial_energy = energies[:, :1]
    drift = (energies - initial_energy).abs() / initial_energy.abs().clamp_min(1e-8)
    return drift.mean()


def discrete_lagrangian(lagrangian, q_prev, q_next, attrs, mask, dt):
    q_mid = 0.5 * (q_prev + q_next)
    q_dot_mid = (q_next - q_prev) / float(dt)
    return float(dt) * lagrangian(q_mid, q_dot_mid, attrs, mask)


def discrete_euler_lagrange_residual(lagrangian, q_prev, q_curr, q_next, attrs, mask, dt):
    q_curr_eval = q_curr if q_curr.requires_grad else q_curr.detach().requires_grad_(True)

    l_prev = discrete_lagrangian(lagrangian, q_prev, q_curr_eval, attrs, mask, dt)
    l_next = discrete_lagrangian(lagrangian, q_curr_eval, q_next, attrs, mask, dt)

    d2_prev = torch.autograd.grad(l_prev.sum(), q_curr_eval, create_graph=True)[0]
    d1_next = torch.autograd.grad(l_next.sum(), q_curr_eval, create_graph=True)[0]
    return d2_prev + d1_next


def del_residual_metric(lagrangian, positions, attrs, mask, dt):
    if positions.shape[1] < 3:
        return positions.new_zeros(())

    mask_traj = _expand_mask(mask, positions.shape[1])
    residual_terms = []
    for step_idx in range(1, positions.shape[1] - 1):
        residual = discrete_euler_lagrange_residual(
            lagrangian=lagrangian,
            q_prev=positions[:, step_idx - 1],
            q_curr=positions[:, step_idx],
            q_next=positions[:, step_idx + 1],
            attrs=attrs,
            mask=mask_traj[:, step_idx],
            dt=dt,
        )
        residual_terms.append(
            masked_position_mse(
                pred=residual.unsqueeze(1),
                target=torch.zeros_like(residual).unsqueeze(1),
                mask=mask_traj[:, step_idx : step_idx + 1],
            )
        )
    return torch.stack(residual_terms).mean()


def trajectory_loss(lagrangian, q0, qd0, attrs, mask, q_gt_traj, dt, n_steps):
    rollout = rollout_trajectory(
        lagrangian=lagrangian,
        q0=q0,
        qd0=qd0,
        attrs=attrs,
        mask=mask if mask.dim() == 2 else mask[:, 0],
        dt=dt,
        n_steps=n_steps,
        return_energies=True,
    )
    pred = rollout["positions"][:, 1 : n_steps + 1]
    target = q_gt_traj[:, 1 : n_steps + 1]
    compare_mask = mask[:, 1 : n_steps + 1] if mask.dim() == 3 else mask
    return masked_position_mse(pred, target, compare_mask), rollout
