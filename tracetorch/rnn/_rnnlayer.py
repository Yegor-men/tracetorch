import torch
import math
from ..core import Layer as BaseLayer
from typing import Literal
from .. import functional


class Layer(BaseLayer):
    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__(num_neurons, dim)
