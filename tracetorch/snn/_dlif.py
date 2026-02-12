from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class DLIF(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.pos_mem = None
        self.neg_mem = None
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)

    @property
    def pos_beta(self):
        return nn.functional.sigmoid(self.raw_pos_beta)

    @property
    def neg_beta(self):
        return nn.functional.sigmoid(self.raw_neg_beta)

    @property
    def threshold(self):
        return nn.functional.softplus(self.raw_threshold)

    def zero_states(self):
        self.pos_mem = None
        self.neg_mem = None

    def detach_states(self):
        if self.pos_mem is not None:
            self.pos_mem = self.pos_mem.detach()
        if self.neg_mem is not None:
            self.neg_mem = self.neg_mem.detach()

    def forward(self, x):
        if self.pos_mem is None:
            self.pos_mem = torch.zeros_like(x)
        if self.neg_mem is None:
            self.neg_mem = torch.zeros_like(x)

        pos_mem_moved = self.pos_mem.movedim(self.dim, -1)
        neg_mem_moved = self.neg_mem.movedim(self.dim, -1)

        x_moved = x.movedim(self.dim, -1)

        pos_mem_moved = pos_mem_moved * self.pos_beta + torch.where(x_moved >= 0, x_moved, 0.0)
        neg_mem_moved = neg_mem_moved * self.neg_beta + torch.where(x_moved < 0, x_moved, 0.0)

        spikes = self.heaviside(pos_mem_moved + neg_mem_moved - self.threshold)

        self.pos_mem = pos_mem_moved.movedim(-1, self.dim)
        self.neg_mem = neg_mem_moved.movedim(-1, self.dim)
        return spikes.movedim(-1, self.dim)
