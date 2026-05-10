import torch
from torch import nn


def sigmoid4x(x: torch.Tensor) -> torch.Tensor:
    r"""Apply the default traceTorch smooth spike function.

    ``sigmoid4x`` is a steeper sigmoid used to turn membrane distance from
    threshold into a differentiable firing probability. SNN firing layers pass
    values such as ``mem - threshold + bias`` through this function before
    applying their ``quant_fn``.

    Args:
        x (torch.Tensor): membrane distance from threshold.

    Returns:
        torch.Tensor: smooth firing probability in ``(0, 1)``.
    """
    return nn.functional.sigmoid(4.0 * x)
