import torch


def compute_acceleration(lagrangian, q, attrs, mask):
    if q.dim() != 3:
        raise ValueError(f"Expected q with shape (B, N, 2), got {tuple(q.shape)}.")

    q_eval = q if q.requires_grad else q.detach().requires_grad_(True)
    potential = lagrangian.potential(q_eval, attrs, mask)
    if not potential.requires_grad:
        return torch.zeros_like(q_eval)
    grad_potential = torch.autograd.grad(
        potential.sum(),
        q_eval,
        create_graph=True,
        allow_unused=True,
    )[0]
    if grad_potential is None:
        grad_potential = torch.zeros_like(q_eval)
    mass = lagrangian.mass(attrs).unsqueeze(-1).clamp_min(1e-8)
    acceleration = -grad_potential / mass
    return acceleration * mask.to(acceleration.dtype).unsqueeze(-1)


def leapfrog_step(lagrangian, q, q_dot, attrs, mask, dt):
    dt = float(dt)
    acceleration_t = compute_acceleration(lagrangian, q, attrs, mask)
    q_dot_half = q_dot + 0.5 * dt * acceleration_t
    q_next = q + dt * q_dot_half

    acceleration_next = compute_acceleration(lagrangian, q_next, attrs, mask)
    q_dot_next = q_dot_half + 0.5 * dt * acceleration_next
    return q_next, q_dot_next


def total_energy(lagrangian, q, q_dot, attrs, mask):
    components = lagrangian.compute_components(q=q, q_dot=q_dot, attrs=attrs, mask=mask)
    return components["kinetic"] + components["potential"]
