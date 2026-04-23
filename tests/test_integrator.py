import torch
import torch.nn as nn

from src.dynamics.integrator import leapfrog_step, total_energy


class FreeParticleLagrangian(nn.Module):
    def mass(self, attrs):
        return torch.ones(attrs.shape[:2], dtype=attrs.dtype, device=attrs.device)

    def potential(self, q, attrs, mask):
        return torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)

    def compute_components(self, q, q_dot, attrs, mask):
        kinetic = 0.5 * (q_dot.pow(2).sum(dim=-1) * mask.to(q.dtype)).sum(dim=1)
        return {
            "kinetic": kinetic,
            "potential": torch.zeros_like(kinetic),
        }


class HarmonicOscillatorLagrangian(nn.Module):
    def __init__(self, stiffness=1.0):
        super().__init__()
        self.stiffness = float(stiffness)

    def mass(self, attrs):
        return torch.ones(attrs.shape[:2], dtype=attrs.dtype, device=attrs.device)

    def potential(self, q, attrs, mask):
        displacement_sq = q.pow(2).sum(dim=-1)
        return 0.5 * self.stiffness * (displacement_sq * mask.to(q.dtype)).sum(dim=1)

    def compute_components(self, q, q_dot, attrs, mask):
        kinetic = 0.5 * (q_dot.pow(2).sum(dim=-1) * mask.to(q.dtype)).sum(dim=1)
        potential = self.potential(q, attrs, mask)
        return {"kinetic": kinetic, "potential": potential}


def test_free_particle_velocity_is_conserved():
    lagrangian = FreeParticleLagrangian()
    attrs = torch.zeros(1, 1, 1, dtype=torch.float64)
    mask = torch.tensor([[True]])
    q = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
    q_dot = torch.tensor([[[1.5, -0.25]]], dtype=torch.float64)

    for _ in range(100):
        q, q_dot = leapfrog_step(lagrangian, q, q_dot, attrs, mask, dt=0.1)

    assert torch.allclose(q_dot, torch.tensor([[[1.5, -0.25]]], dtype=torch.float64), atol=1e-9)


def test_harmonic_oscillator_energy_stays_bounded():
    lagrangian = HarmonicOscillatorLagrangian(stiffness=1.0)
    attrs = torch.zeros(1, 1, 1, dtype=torch.float64)
    mask = torch.tensor([[True]])
    q = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    q_dot = torch.tensor([[[0.0, 0.5]]], dtype=torch.float64)

    initial_energy = total_energy(lagrangian, q, q_dot, attrs, mask)
    max_relative_drift = 0.0
    for _ in range(1000):
        q, q_dot = leapfrog_step(lagrangian, q, q_dot, attrs, mask, dt=0.05)
        energy = total_energy(lagrangian, q, q_dot, attrs, mask)
        drift = ((energy - initial_energy).abs() / initial_energy.abs().clamp_min(1e-8)).item()
        max_relative_drift = max(max_relative_drift, drift)

    assert max_relative_drift < 0.02
