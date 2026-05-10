import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def create_diverging_colormap():
    r"""Create the default diverging colormap for signed spike values.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: blue-white-red colormap where
        negative values are blue, zero is white, and positive values are red.
    """
    colors = ['blue', 'white', 'red']  # negative -> white -> positive
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('diverging', colors, N=n_bins)
    return cmap


def spike_train(
        list_of_tensors,
        spacing: float = 1.0,
        linelength: float = 0.8,
        linewidth: float = 0.5,
        title: str = "Spike Train Raster",
        use_imshow: bool = True,
):
    r"""Plot a spike train or signed activity sequence.

    Args:
        list_of_tensors (Sequence[torch.Tensor]): sequence of tensors, one per
            timestep. Each tensor is flattened as neuron/activity values.
        spacing (float, default=1.0): vertical spacing for event-plot mode.
        linelength (float, default=0.8): line length for event-plot mode.
        linewidth (float, default=0.5): line width for event-plot mode.
        title (str, default="Spike Train Raster"): plot title.
        use_imshow (bool, default=True): if True, draw a signed heatmap. If
            False, draw an event plot of nonzero entries.

    Notes:
        This helper is intended for quick experiment visualization. It calls
        ``plt.show()`` and does not return the figure.
    """
    data = torch.stack(list_of_tensors).cpu().detach().numpy()  # shape (T, N)
    data = data.T  # shape (N, T)

    plt.figure(figsize=(8, 4))
    if use_imshow:
        # Create custom colormap for diverging values
        cmap = create_diverging_colormap()

        # Determine symmetric color scale
        vmax = max(abs(data.max()), abs(data.min()))
        vmin = -vmax

        plt.imshow(
            data,
            aspect='auto',
            cmap=cmap,
            origin='lower',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(label='Spike value')
    else:
        N, T = data.shape
        # for each neuron i, list of spike times where value != 0
        spike_times = [
            [t for t in range(T) if data[i, t] != 0]
            for i in range(N)
        ]
        # y offsets
        offsets = np.arange(N) * spacing
        plt.eventplot(
            spike_times,
            orientation='horizontal',
            lineoffsets=offsets,
            linelengths=linelength,
            linewidths=linewidth,
            colors='k'
        )
        plt.ylim(-spacing, offsets[-1] + spacing)

    plt.xlabel("Time step")
    plt.ylabel("Neuron index")
    plt.title(title)
    plt.tight_layout()
    plt.show()
