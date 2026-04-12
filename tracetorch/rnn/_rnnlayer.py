import torch
import math
from ..core import Layer as BaseLayer
from typing import Literal
from .. import functional


class Layer(BaseLayer):
    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__(num_neurons, dim)

    # def _register_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
    #     self._register_parameter(
    #         name, value, rank, learnable,
    #         inverse_function=functional.mamba_scale,
    #         activation_function=torch.exp,
    #     )
