import torch
from torch import nn


def sigmoid4x(x: torch.Tensor) -> torch.Tensor:
    return nn.functional.sigmoid(4.0 * x)
