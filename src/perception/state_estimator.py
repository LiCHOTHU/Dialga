import torch


def compute_mask_centroids(masks):
    """
    masks: (T, N, H, W) bool or float tensor.
    returns: (T, N, 2) tensor of (x, y) centroids in pixel coordinates.
    """
    if masks.dim() != 4:
        raise ValueError(f"Expected masks with shape (T, N, H, W), got {tuple(masks.shape)}.")

    masks_float = masks.to(torch.float32)
    time_steps, num_objects, height, width = masks.shape
    ys = torch.arange(height, device=masks.device, dtype=torch.float32).view(1, 1, height, 1)
    xs = torch.arange(width, device=masks.device, dtype=torch.float32).view(1, 1, 1, width)

    mass = masks_float.sum(dim=(-1, -2)).clamp_min(1e-6)
    centroid_x = (masks_float * xs).sum(dim=(-1, -2)) / mass
    centroid_y = (masks_float * ys).sum(dim=(-1, -2)) / mass
    centroids = torch.stack([centroid_x, centroid_y], dim=-1)

    empty_masks = masks_float.sum(dim=(-1, -2)) <= 0
    centroids[empty_masks] = 0.0
    return centroids


def masks_to_states(masks, dt=1.0):
    q = compute_mask_centroids(masks)
    q_dot = torch.zeros_like(q)
    if q.shape[0] >= 2:
        q_dot[1:-1] = (q[2:] - q[:-2]) / (2.0 * float(dt))
        q_dot[0] = (q[1] - q[0]) / float(dt)
        q_dot[-1] = (q[-1] - q[-2]) / float(dt)
    return q, q_dot
