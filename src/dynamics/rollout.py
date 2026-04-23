import torch

from .integrator import leapfrog_step, total_energy


def rollout_trajectory(
    lagrangian,
    q0,
    qd0,
    attrs,
    mask,
    dt,
    n_steps,
    return_energies=True,
):
    q = q0
    q_dot = qd0

    positions = [q0]
    velocities = [qd0]
    energies = [total_energy(lagrangian, q0, qd0, attrs, mask)] if return_energies else None

    for _ in range(int(n_steps)):
        q, q_dot = leapfrog_step(
            lagrangian=lagrangian,
            q=q,
            q_dot=q_dot,
            attrs=attrs,
            mask=mask,
            dt=dt,
        )
        positions.append(q)
        velocities.append(q_dot)
        if return_energies:
            energies.append(total_energy(lagrangian, q, q_dot, attrs, mask))

    rollout = {
        "positions": torch.stack(positions, dim=1),
        "velocities": torch.stack(velocities, dim=1),
    }
    if return_energies:
        rollout["energies"] = torch.stack(energies, dim=1)
    return rollout
