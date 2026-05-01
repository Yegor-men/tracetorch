import torch
import math
from ..core import Layer as BaseLayer
from typing import Literal
from .. import functional


class Layer(BaseLayer):
    r"""The superclass used for all RNN layers.

    Args:
        num_neurons (int): the number of neurons the layer is considered to have. When initializing any hidden states or registering parameters via the tracetorch methods, this is the value used.
        dim (int, default=-1): the dimension along which the layer operates.

    Notes:
        Inherits from ``tt.core.Layer``, but doesn't add any new methods. Check ``tt.core.Layer`` to see available methods.
    """

    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__(num_neurons, dim)
