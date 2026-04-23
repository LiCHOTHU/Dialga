from pathlib import Path


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization utilities.") from exc


def plot_2d_trajectories(positions, mask=None, save_path=None, title="Trajectories"):
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    positions_cpu = positions.detach().cpu()
    num_objects = positions_cpu.shape[1]
    for obj_idx in range(num_objects):
        if mask is not None and not bool(mask[..., obj_idx].any().item()):
            continue
        ax.plot(positions_cpu[:, obj_idx, 0], positions_cpu[:, obj_idx, 1], marker="o")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
