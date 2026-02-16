from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ..snn._ttmodule import TTModule
from ..snn._layer_mixin import LayerMixin
from .. import functional


class LITS(TTModule, LayerMixin):
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
        TTModule.__init__(self)
        LayerMixin.__init__(self, num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def zero_states(self):
        self._zero_states()

    def detach_states(self):
        self._detach_states()

    def forward(self, x):
        self._ensure_states(x)

        x_moved = self._to_working_dim(x)

        mem_moved = self._to_working_dim(self.mem)
        mem_moved = mem_moved * self.beta + x_moved

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes_moved = pos_spikes + neg_spikes
        spikes = self._from_working_dim(spikes_moved)

        self.mem = self._from_working_dim(mem_moved)

        return spikes


class DLITS(TTModule, LayerMixin):
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
        TTModule.__init__(self)
        LayerMixin.__init__(self, num_neurons, dim)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def zero_states(self):
        self._zero_states()

    def detach_states(self):
        self._detach_states()

    def forward(self, x):
        self._ensure_states(x)

        x_moved = self._to_working_dim(x)

        pos_mem_moved = self._to_working_dim(self.pos_mem)
        neg_mem_moved = self._to_working_dim(self.neg_mem)

        pos_mem_moved = pos_mem_moved * self.pos_beta + torch.where(x_moved >= 0, x_moved, 0.0)
        neg_mem_moved = neg_mem_moved * self.neg_beta + torch.where(x_moved < 0, x_moved, 0.0)

        mem_moved = pos_mem_moved + neg_mem_moved

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes_moved = pos_spikes + neg_spikes
        spikes = self._from_working_dim(spikes_moved)

        self.pos_mem = self._from_working_dim(pos_mem_moved)
        self.neg_mem = self._from_working_dim(neg_mem_moved)

        return spikes


class SLITS(TTModule, LayerMixin):
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
        TTModule.__init__(self)
        LayerMixin.__init__(self, num_neurons, dim)

        self._initialize_state("syn")
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.heaviside = surrogate_derivative
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def zero_states(self):
        self._zero_states()

    def detach_states(self):
        self._detach_states()

    def forward(self, x):
        self._ensure_states(x)

        x_moved = self._to_working_dim(x)

        syn_moved = self._to_working_dim(self.syn)
        syn_moved = syn_moved * self.alpha + x_moved * (1 - self.alpha)

        mem_moved = self._to_working_dim(self.mem)
        mem_moved = mem_moved * self.beta + syn_moved * (1 - self.beta)

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes_moved = pos_spikes + neg_spikes
        spikes = self._from_working_dim(spikes_moved)

        self.syn = self._from_working_dim(syn_moved)
        self.mem = self._from_working_dim(mem_moved)

        return spikes


class RLITS(TTModule, LayerMixin):
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
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_rec_weight: bool = True,
            learn_bias: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        TTModule.__init__(self)
        LayerMixin.__init__(self, num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self._initialize_state("rec")
        self._initialize_state("prev_output")
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.heaviside = surrogate_derivative
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)
        self._register_parameter("bias", bias, bias_rank, learn_bias)

    def zero_states(self):
        self._zero_states()

    def detach_states(self):
        self._detach_states()

    def forward(self, x):
        self._ensure_states(x)

        x_moved = self._to_working_dim(x)

        rec_moved = self._to_working_dim(self.rec)
        prev_output_moved = self._to_working_dim(self.prev_output)
        rec_moved = rec_moved * self.gamma + prev_output_moved * (1 - self.gamma)

        mem_delta = rec_moved * self.rec_weight + x_moved + self.bias

        mem_moved = self._to_working_dim(self.mem)
        mem_moved = mem_moved * self.beta + mem_delta

        pos_spikes = self.heaviside(mem_moved - self.pos_threshold) * self.pos_scale
        neg_spikes = -self.heaviside(self.neg_threshold - mem_moved) * self.neg_scale
        spikes_moved = pos_spikes + neg_spikes
        spikes = self._from_working_dim(spikes_moved)

        self.mem = self._from_working_dim(mem_moved)
        self.rec = self._from_working_dim(rec_moved)
        self.prev_output = spikes

        return spikes
