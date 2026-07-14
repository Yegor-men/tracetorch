from ..core import Layer as BaseLayer
from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional


class Layer(BaseLayer):
    r"""Base class for traceTorch SNN layers.

    This class extends ``tt.Layer`` with parameter registration helpers commonly
    used by spiking layers:

    * decays are constrained to ``(0, 1)`` through a sigmoid transform;
    * thresholds are constrained to positive values through a softplus transform;
    * biases use a smooth unconstrained transform.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        dim (int, default=-1): dimension along which the layer operates.

    Notes:
        Users normally instantiate concrete layers such as ``LIB`` or ``LIT``.
        Subclass this base when creating a custom SNN layer that should integrate
        with ``tt.Model`` state management and traceTorch parameter compilation.
    """

    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__(num_neurons, dim)

    def define_decay(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        r"""Register a decay parameter constrained to ``(0, 1)``."""
        self.define_parameter(
            name,
            value,
            rank,
            learnable,
            init_fn=functional.sigmoid_inverse,
            inverse_fn=functional.sigmoid_inverse,
            activation_fn=nn.functional.sigmoid,
        )

    def define_threshold(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        r"""Register a positive threshold parameter."""
        self.define_parameter(
            name,
            value,
            rank,
            learnable,
            init_fn=functional.softplus_inverse,
            inverse_fn=functional.softplus_inverse,
            activation_fn=nn.functional.softplus,
        )
