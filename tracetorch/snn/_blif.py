from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class BLIF(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    @property
    def pos_threshold(self):
        return nn.functional.softplus(self.raw_pos_threshold)

    @property
    def neg_threshold(self):
        return nn.functional.softplus(self.raw_neg_threshold)

    @property
    def pos_scale(self):
        return self.raw_pos_scale

    @property
    def neg_scale(self):
        return self.raw_neg_scale

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

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes = pos_spikes + neg_spikes

        self.mem = mem_moved.movedim(-1, self.dim)
        return spikes.movedim(-1, self.dim)
