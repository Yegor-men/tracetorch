from typing import Tuple, Union

import torch
from torch import nn


class MoveDim(nn.Module):
    """A ``torch.movedim`` wrapper for explicit layout changes in Sequential.

    For channel-first images, place ``MoveDim(-3, -1)`` before a traceTorch
    layer and ``MoveDim(-1, -3)`` after it. States then stay feature-last too.
    """

    def __init__(self, source: Union[int, Tuple[int, ...]],
                 destination: Union[int, Tuple[int, ...]]):
        super().__init__()
        self.source = source
        self.destination = destination

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.movedim(self.source, self.destination)

    def extra_repr(self) -> str:
        return f"source={self.source}, destination={self.destination}"
