import torch


class CompositeRenderer:
    """
    Lightweight object compositor for trajectory visualization.

    This renderer assumes a static background and translates frame-0 object crops
    by integer centroid offsets.
    """

    def render(self, frame0, masks0, q_traj, q0=None):
        if frame0.dim() != 3:
            raise ValueError(f"Expected frame0 with shape (C, H, W), got {tuple(frame0.shape)}.")
        if masks0.dim() != 3:
            raise ValueError(f"Expected masks0 with shape (N, H, W), got {tuple(masks0.shape)}.")
        if q_traj.dim() != 3:
            raise ValueError(f"Expected q_traj with shape (T, N, 2), got {tuple(q_traj.shape)}.")

        num_objects, height, width = masks0.shape
        if q0 is None:
            q0 = self._mask_centroids(masks0)

        background = frame0.clone()
        foreground = []
        alpha = []
        for obj_idx in range(num_objects):
            obj_mask = masks0[obj_idx : obj_idx + 1].to(frame0.dtype)
            background = background * (1.0 - obj_mask)
            foreground.append(frame0 * obj_mask)
            alpha.append(obj_mask)

        frames = []
        for step_idx in range(q_traj.shape[0]):
            frame = background.clone()
            for obj_idx in range(num_objects):
                shift = (q_traj[step_idx, obj_idx] - q0[obj_idx]).round().to(torch.int64)
                shifted_rgb = torch.roll(foreground[obj_idx], shifts=(int(shift[1]), int(shift[0])), dims=(-2, -1))
                shifted_alpha = torch.roll(alpha[obj_idx], shifts=(int(shift[1]), int(shift[0])), dims=(-2, -1))
                frame = shifted_rgb * shifted_alpha + frame * (1.0 - shifted_alpha)
            frames.append(frame)
        return torch.stack(frames, dim=0)

    @staticmethod
    def _mask_centroids(masks):
        masks_float = masks.to(torch.float32)
        num_objects, height, width = masks.shape
        ys = torch.arange(height, device=masks.device, dtype=torch.float32).view(1, height, 1)
        xs = torch.arange(width, device=masks.device, dtype=torch.float32).view(1, 1, width)
        area = masks_float.sum(dim=(-1, -2)).clamp_min(1e-6)
        centroid_x = (masks_float * xs).sum(dim=(-1, -2)) / area
        centroid_y = (masks_float * ys).sum(dim=(-1, -2)) / area
        return torch.stack([centroid_x, centroid_y], dim=-1)
