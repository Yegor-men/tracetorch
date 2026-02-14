from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class LIB(TTModule, SetupMixin):
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


class DLIB(TTModule, SetupMixin):
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


class SLIB(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        self.syn = None
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)

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
        mem_moved = mem_moved * self.beta + syn_moved

        spikes = self.heaviside(mem_moved - self.threshold)

        self.syn = syn_moved.movedim(-1, self.dim)
        self.mem = mem_moved.movedim(-1, self.dim)
        return spikes.movedim(-1, self.dim)


class RLIB(TTModule, SetupMixin):
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
