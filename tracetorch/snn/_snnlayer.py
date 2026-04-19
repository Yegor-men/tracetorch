from ..core import Layer as BaseLayer
from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional


class Layer(BaseLayer):
    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__(num_neurons, dim)

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
            init_fn=functional.sigmoid_inverse,
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
            init_fn=functional.softplus_inverse,
            inverse_function=functional.softplus_inverse,
            activation_function=nn.functional.softplus,
        )

    def _register_bias(
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
            init_fn=torch.sinh,
            inverse_function=torch.sinh,
            activation_function=torch.arcsinh,
        )
