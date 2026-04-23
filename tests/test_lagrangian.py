import torch

from src.dynamics.lagrangian import ObjectLagrangian


def test_permutation_equivariance():
    torch.manual_seed(0)
    model = ObjectLagrangian(attr_dim=13, hidden_size=32)
    q = torch.randn(2, 4, 2)
    q_dot = torch.randn(2, 4, 2)
    attrs = torch.randn(2, 4, 13)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    lagrangian_a = model(q, q_dot, attrs, mask)
    permutation = torch.tensor([2, 0, 1, 3])
    lagrangian_b = model(
        q[:, permutation],
        q_dot[:, permutation],
        attrs[:, permutation],
        mask[:, permutation],
    )

    assert torch.allclose(lagrangian_a, lagrangian_b, atol=1e-5)
