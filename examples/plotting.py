import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


def _as_detached_cpu(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    return torch.as_tensor(tensor).detach().cpu()


def _to_image_batch(tensor):
    tensor = _as_detached_cpu(tensor)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] in (1, 3):
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected image tensor with 2, 3, or 4 dimensions, got {tuple(tensor.shape)}")

    if tensor.shape[1] not in (1, 3):
        raise ValueError(f"Expected grayscale or RGB images with 1 or 3 channels, got {tensor.shape[1]}")

    return tensor


def plot_image_grid(
    tensor,
    *,
    title=None,
    max_images=32,
    columns=None,
    cmap="gray",
    vmin=None,
    vmax=None,
    save_path=None,
    show=False,
):
    """Plot a batch of grayscale or RGB images.

    Accepts ``[H, W]``, ``[C, H, W]``, ``[B, H, W]``, or ``[B, C, H, W]``.
    Returns ``(fig, axes)`` so examples can decide when to show or save.
    """
    images = _to_image_batch(tensor)[:max_images]
    batch, channels, _, _ = images.shape

    if columns is None:
        columns = math.ceil(math.sqrt(batch))
    rows = math.ceil(batch / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.0), squeeze=False)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= batch:
            continue

        image = images[idx]
        if channels == 1:
            ax.imshow(image.squeeze(0), cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1))

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig, axes


def _activity_matrix(activity):
    if isinstance(activity, torch.Tensor):
        data = activity.detach().cpu()
    else:
        data = torch.stack([_as_detached_cpu(step) for step in activity], dim=0)

    if data.ndim < 2:
        raise ValueError(f"Expected activity with shape [T, ...], got {tuple(data.shape)}")

    return data.flatten(start_dim=1).T.numpy()


def _diverging_cmap():
    return LinearSegmentedColormap.from_list("tracetorch_activity", ["#2563eb", "#ffffff", "#dc2626"], N=256)


def plot_spike_train(
    activity,
    *,
    title="Activity over time",
    heatmap=True,
    symmetric=True,
    spacing=1.0,
    linelength=0.8,
    linewidth=0.5,
    cmap=None,
    colorbar_label="value",
    show=False,
):
    """Plot timestep activity as a signed heatmap or event raster.

    ``activity`` can be a tensor shaped ``[T, ...]`` or a list of timestep
    tensors. Every non-time dimension is flattened into the neuron axis.
    """
    data = _activity_matrix(activity)
    fig, ax = plt.subplots(figsize=(9, 4))

    if heatmap:
        if cmap is None:
            cmap = _diverging_cmap()
        if symmetric:
            vmax = float(np.max(np.abs(data))) if data.size else 0.0
            vmax = vmax if vmax > 0 else 1.0
            vmin = -vmax
        else:
            vmin = float(np.min(data)) if data.size else 0.0
            vmax = float(np.max(data)) if data.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
        image = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax, label=colorbar_label)
    else:
        neuron_count, timestep_count = data.shape
        spike_times = [
            [t for t in range(timestep_count) if data[neuron_idx, t] != 0]
            for neuron_idx in range(neuron_count)
        ]
        offsets = np.arange(neuron_count) * spacing
        ax.eventplot(
            spike_times,
            orientation="horizontal",
            lineoffsets=offsets,
            linelengths=linelength,
            linewidths=linewidth,
            colors="black",
        )
        if neuron_count > 0:
            ax.set_ylim(-spacing, offsets[-1] + spacing)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_value_histograms(
    tensors,
    *,
    labels=None,
    title="Value distributions",
    bins=100,
    density=True,
    alpha=0.45,
    show=False,
):
    """Plot simple value histograms for one or more tensors.

    This is intentionally lighter than the old KDE distribution helper. It is
    useful for quick example diagnostics without bringing more API surface into
    the package itself.
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    if labels is None:
        labels = [f"tensor {idx}" for idx in range(len(tensors))]
    if len(labels) != len(tensors):
        raise ValueError("labels must have the same length as tensors")

    fig, ax = plt.subplots(figsize=(8, 4))
    for tensor, label in zip(tensors, labels):
        values = _as_detached_cpu(tensor).flatten().numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, density=density, alpha=alpha, label=label)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize="small")
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax
