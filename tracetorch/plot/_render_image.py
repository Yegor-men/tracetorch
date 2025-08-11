import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def render_image(tensor, figsize=(4, 4), title=None, vmin=None, vmax=None):
    """
    Plot a tensor image.

    - tensor: torch.Tensor in either HW (H,W) or CHW (C,H,W) format.
              C may be 1 (grayscale) or 3 (RGB).
    - figsize: matplotlib figure size tuple
    - title: optional title string
    - vmin/vmax: if provided, these define the original data range (used for the colorbar).
                 If None, the function uses tensor.min() / tensor.max().
    Behavior change for grayscale: the image is rescaled to [0,1] for display (max contrast),
    but the colorbar shows the original numeric range (vmin..vmax).
    :param tensor:
    :param figsize:
    :param title:
    :param vmin:
    :param vmax:
    :return:
    """
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)

    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()

    # helper to determine original vmin/vmax (either from args or data)
    def _orig_range(arr, vmin_arg, vmax_arg):
        data_min = float(np.nanmin(arr))
        data_max = float(np.nanmax(arr))
        orig_vmin = data_min if vmin_arg is None else float(vmin_arg)
        orig_vmax = data_max if vmax_arg is None else float(vmax_arg)
        return orig_vmin, orig_vmax

    if t.dim() == 2:
        # HW grayscale
        img = t.numpy().astype(np.float32)

        orig_vmin, orig_vmax = _orig_range(img, vmin, vmax)

        # avoid divide-by-zero for constant images
        if orig_vmax > orig_vmin:
            img_scaled = (img - orig_vmin) / (orig_vmax - orig_vmin)
            # numerical safety
            img_scaled = np.clip(img_scaled, 0.0, 1.0)
        else:
            # constant image -> show mid-gray but colorbar will show the constant value
            img_scaled = np.full_like(img, 0.5, dtype=np.float32)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(img_scaled, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.axis("off")
        if title:
            ax.set_title(title)

        # colorbar referencing original scale
        mappable = cm.ScalarMappable(norm=Normalize(vmin=orig_vmin, vmax=orig_vmax), cmap="gray")
        mappable.set_array([])  # required for colorbar
        plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        plt.show()
        return

    if t.dim() == 3:
        c, h, w = t.shape
        if c == 1:
            img = t.squeeze(0).numpy().astype(np.float32)

            orig_vmin, orig_vmax = _orig_range(img, vmin, vmax)

            if orig_vmax > orig_vmin:
                img_scaled = (img - orig_vmin) / (orig_vmax - orig_vmin)
                img_scaled = np.clip(img_scaled, 0.0, 1.0)
            else:
                img_scaled = np.full_like(img, 0.5, dtype=np.float32)

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(img_scaled, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.axis("off")
            if title:
                ax.set_title(title)

            mappable = cm.ScalarMappable(norm=Normalize(vmin=orig_vmin, vmax=orig_vmax), cmap="gray")
            mappable.set_array([])
            plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            plt.show()
            return

        elif c == 3:
            # CHW -> HWC for matplotlib (leave RGB untouched)
            img = t.permute(1, 2, 0).numpy()
            img = np.clip(img, 0.0, 1.0)  # assume RGB in 0..1; you can change handling if needed
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            if title:
                ax.set_title(title)
            plt.show()
            return
        else:
            raise ValueError(f"Unsupported channel dimension: C={c}. Expected C==1 or C==3 for CHW.")

    raise ValueError(f"Unsupported tensor shape {tuple(t.shape)}. Expected HW or CHW.")
