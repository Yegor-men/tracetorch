from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._param_setup import SetupMixin
from .. import functional


class LITS(TTModule, SetupMixin):
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


class DLITS(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
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
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    @property
    def pos_beta(self):
        return nn.functional.sigmoid(self.raw_pos_beta)

    @property
    def neg_beta(self):
        return nn.functional.sigmoid(self.raw_neg_beta)

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

        mem_moved = pos_mem_moved + neg_mem_moved

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes = pos_spikes + neg_spikes

        self.pos_mem = pos_mem_moved.movedim(-1, self.dim)
        self.neg_mem = neg_mem_moved.movedim(-1, self.dim)
        return spikes.movedim(-1, self.dim)


class SLITS(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
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

        self.syn = None
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.mem = None
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    @property
    def alpha(self):
        return nn.functional.sigmoid(self.raw_alpha)

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

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes = pos_spikes + neg_spikes

        self.syn = syn_moved.movedim(-1, self.dim)
        self.mem = mem_moved.movedim(-1, self.dim)
        return spikes.movedim(-1, self.dim)


class RLITS(TTModule, SetupMixin):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
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
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    @property
    def gamma(self):
        return nn.functional.sigmoid(self.raw_gamma)

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

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes = (pos_spikes + neg_spikes).movedim(-1, self.dim)

        self.mem = mem_moved.movedim(-1, self.dim)
        self.rec = rec_moved.movedim(-1, self.dim)
        self.prev_output = spikes
        return spikes
