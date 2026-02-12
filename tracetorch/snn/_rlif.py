from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class RLIF(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            learn_gamma: bool = True,
            learn_rec_weight: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.rec = None
        self.prev_output = None
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.heaviside = surrogate_derivative
        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    @property
    def gamma(self):
        return nn.functional.sigmoid(self.raw_gamma)

    @property
    def threshold(self):
        return nn.functional.softplus(self.raw_threshold)

    @property
    def rec_weight(self):
        return self.raw_rec_weight

    def zero_states(self):
        self.mem = None
        self.rec = None
        self.prev_output = None

    def detach_states(self):
        if self.mem is not None:
            self.mem = self.mem.detach()
        if self.rec is not None:
            self.rec = self.rec.detach()
        if self.prev_output is not None:
            self.prev_output = self.prev_output.detach()

    def forward(self, x):
        if self.rec is None:
            self.rec = torch.zeros_like(x)
        if self.prev_output is None:
            self.prev_output = torch.zeros_like(x)

        rec_moved = self.rec.movedim(self.dim, -1)
        rec_moved = rec_moved * self.gamma + self.prev_output.movedim(self.dim, -1) * (1 - self.gamma)

        if self.mem is None:
            self.mem = torch.zeros_like(x)

        mem_moved = self.mem.movedim(self.dim, -1)
        mem_moved = mem_moved * self.beta + rec_moved * self.rec_weight + x.movedim(self.dim, -1)

        spikes = self.heaviside(mem_moved - self.threshold)

        self.mem = mem_moved.movedim(-1, self.dim)
        self.rec = rec_moved.movedim(-1, self.dim)
        output = spikes.movedim(-1, self.dim)
        self.prev_output = output
        return output
