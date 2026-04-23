import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectLagrangian(nn.Module):
    """
    Permutation-invariant object Lagrangian over 2D states.

    The public CLEVRER annotations expose 3D object locations but not camera
    projection metadata. In the current pipeline, the dataset chooses a 2D slice
    of those coordinates, and this module operates on that 2D state.
    """

    def __init__(self, attr_dim, hidden_size=128, mass_floor=1e-2):
        super().__init__()
        self.attr_dim = int(attr_dim)
        self.hidden_size = int(hidden_size)
        self.mass_floor = float(mass_floor)

        self.mass_net = nn.Sequential(
            nn.Linear(self.attr_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.V_ext = nn.Sequential(
            nn.Linear(2 + self.attr_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.V_pair = nn.Sequential(
            nn.Linear(2 + 2 * self.attr_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def mass(self, attrs):
        return F.softplus(self.mass_net(attrs).squeeze(-1)) + self.mass_floor

    def potential(self, q, attrs, mask):
        components = self.compute_components(q=q, q_dot=None, attrs=attrs, mask=mask)
        return components["potential"]

    def kinetic(self, q_dot, attrs, mask):
        mask_float = mask.to(q_dot.dtype)
        mass = self.mass(attrs)
        speed_sq = q_dot.pow(2).sum(dim=-1)
        return 0.5 * (mass * speed_sq * mask_float).sum(dim=1)

    def compute_components(self, q, q_dot, attrs, mask):
        if q.dim() != 3 or q.shape[-1] != 2:
            raise ValueError(f"Expected q with shape (B, N, 2), got {tuple(q.shape)}.")
        if attrs.dim() != 3:
            raise ValueError(
                f"Expected attrs with shape (B, N, d_attr), got {tuple(attrs.shape)}."
            )
        if mask.dim() != 2:
            raise ValueError(f"Expected mask with shape (B, N), got {tuple(mask.shape)}.")

        batch_size, num_objects, _ = q.shape
        mask_float = mask.to(q.dtype)

        ext_in = torch.cat([q, attrs], dim=-1)
        per_object_external = self.V_ext(ext_in).squeeze(-1)
        external = (per_object_external * mask_float).sum(dim=1)

        q_i = q.unsqueeze(2).expand(batch_size, num_objects, num_objects, 2)
        q_j = q.unsqueeze(1).expand(batch_size, num_objects, num_objects, 2)
        rel_ij = q_j - q_i

        attrs_i = attrs.unsqueeze(2).expand(batch_size, num_objects, num_objects, self.attr_dim)
        attrs_j = attrs.unsqueeze(1).expand(batch_size, num_objects, num_objects, self.attr_dim)

        pair_input_ij = torch.cat([rel_ij, attrs_i, attrs_j], dim=-1)
        pair_input_ji = torch.cat([-rel_ij, attrs_j, attrs_i], dim=-1)

        # Averaging both directions makes the unordered pair term invariant to
        # object index permutations.
        per_pair = 0.5 * (
            self.V_pair(pair_input_ij).squeeze(-1)
            + self.V_pair(pair_input_ji).squeeze(-1)
        )

        pair_mask = mask_float.unsqueeze(2) * mask_float.unsqueeze(1)
        upper_triangle = torch.triu(
            torch.ones(num_objects, num_objects, device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        pair_mask = pair_mask * upper_triangle.unsqueeze(0)
        pairwise = (per_pair * pair_mask).sum(dim=(1, 2))

        potential = external + pairwise
        mass = self.mass(attrs)

        if q_dot is None:
            kinetic = q.new_zeros(batch_size)
            lagrangian = -potential
            speed_sq = q.new_zeros(batch_size, num_objects)
        else:
            speed_sq = q_dot.pow(2).sum(dim=-1)
            kinetic = 0.5 * (mass * speed_sq * mask_float).sum(dim=1)
            lagrangian = kinetic - potential

        return {
            "lagrangian": lagrangian,
            "kinetic": kinetic,
            "potential": potential,
            "mass": mass,
            "per_object_external": per_object_external,
            "per_pair": per_pair,
            "speed_sq": speed_sq,
        }

    def forward(self, q, q_dot, attrs, mask):
        return self.compute_components(q=q, q_dot=q_dot, attrs=attrs, mask=mask)[
            "lagrangian"
        ]
