from ..core import Layer as BaseLayer
from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional


class Layer(BaseLayer):
    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__(num_neurons, dim)
        self.round_ste = functional.round_ste()
        self.bernoulli_ste = functional.bernoulli_ste()
        self.probabilistic_ste = functional.probabilistic_ste()

    def _register_decay(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        self._register_parameter(
            name,
            value,
            rank,
            learnable,
            inverse_function=functional.sigmoid_inverse,
            activation_function=nn.functional.sigmoid,
        )

    def _register_threshold(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        self._register_parameter(
            name,
            value,
            rank,
            learnable,
            inverse_function=functional.softplus_inverse,
            activation_function=nn.functional.softplus,
        )
