from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class LIF(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    @property
    def threshold(self):
        return nn.functional.softplus(self.raw_threshold)

    def zero_states(self):
        self.mem = None

    def detach_states(self):
        if self.mem is not None:
            self.mem = self.mem.detach()

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)

        mem_moved = self.mem.movedim(self.dim, -1)
        mem_moved = mem_moved * self.beta + x.movedim(self.dim, -1)

        spikes = self.heaviside(mem_moved - self.threshold)

        self.mem = mem_moved.movedim(-1, self.dim)
        return spikes.movedim(-1, self.dim)
