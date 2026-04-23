import torch
import torch.nn as nn

from src.dynamics.integrator import total_energy
from src.dynamics.rollout import rollout_trajectory


class QuadraticPotentialLagrangian(nn.Module):
    def mass(self, attrs):
        return torch.ones(attrs.shape[:2], dtype=attrs.dtype, device=attrs.device)

    def potential(self, q, attrs, mask):
        return 0.5 * (q.pow(2).sum(dim=-1) * mask.to(q.dtype)).sum(dim=1)

    def compute_components(self, q, q_dot, attrs, mask):
        kinetic = 0.5 * (q_dot.pow(2).sum(dim=-1) * mask.to(q.dtype)).sum(dim=1)
        potential = self.potential(q, attrs, mask)
        return {"kinetic": kinetic, "potential": potential}


def test_rollout_energy_drift_is_small_for_quadratic_system():
    lagrangian = QuadraticPotentialLagrangian()
    attrs = torch.zeros(1, 2, 1, dtype=torch.float64)
    mask = torch.tensor([[True, True]])
    q0 = torch.tensor([[[1.0, 0.0], [0.0, 0.5]]], dtype=torch.float64)
    qd0 = torch.tensor([[[0.0, 0.5], [-0.25, 0.0]]], dtype=torch.float64)

    rollout = rollout_trajectory(
        lagrangian=lagrangian,
        q0=q0,
        qd0=qd0,
        attrs=attrs,
        mask=mask,
        dt=0.05,
        n_steps=400,
        return_energies=True,
    )

    initial = rollout["energies"][:, :1]
    drift = (rollout["energies"] - initial).abs() / initial.abs().clamp_min(1e-8)
    assert drift.max().item() < 0.02

    final_energy = total_energy(
        lagrangian,
        rollout["positions"][:, -1],
        rollout["velocities"][:, -1],
        attrs,
        mask,
    )
    assert torch.allclose(final_energy, rollout["energies"][:, -1], atol=1e-10)
