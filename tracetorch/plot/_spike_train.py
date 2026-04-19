import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def create_diverging_colormap():
    """Create a colormap with red for positive, blue for negative values."""
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
    # Stack into 2D array: shape (num_neurons, T)
    # list_of_tensors assumed length T each
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
