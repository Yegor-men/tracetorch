from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class LI(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    def zero_states(self):
        self.mem = None

    def detach_states(self):
        if self.mem is not None:
            self.mem = self.mem.detach()

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)

        mem_moved = self.mem.movedim(self.dim, -1)
        mem_moved = mem_moved * self.beta + x.movedim(self.dim, -1) * (1 - self.beta)

        self.mem = mem_moved.movedim(-1, self.dim)
        return self.mem


class SLI(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.syn = None
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

    @property
    def alpha(self):
        return nn.functional.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    @property
    def threshold(self):
        return nn.functional.softplus(self.raw_threshold)

    def zero_states(self):
        self.syn = None
        self.mem = None

    def detach_states(self):
        if self.syn is not None:
            self.syn = self.syn.detach()
        if self.mem is not None:
            self.mem = self.mem.detach()

    def forward(self, x):
        if self.syn is None:
            self.syn = torch.zeros_like(x)

        syn_moved = self.syn.movedim(self.dim, -1)
        syn_moved = syn_moved * self.alpha + x.movedim(self.dim, -1) * (1 - self.alpha)

        if self.mem is None:
            self.mem = torch.zeros_like(x)

        mem_moved = self.mem.movedim(self.dim, -1)
        mem_moved = mem_moved * self.beta + syn_moved * (1 - self.beta)

        self.syn = syn_moved.movedim(-1, self.dim)
        self.mem = mem_moved.movedim(-1, self.dim)
        return self.mem
