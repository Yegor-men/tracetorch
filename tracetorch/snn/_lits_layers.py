from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ._tt_infrastructure import TTLayer
from .. import functional


class LITS(TTLayer):
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
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + x

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        mem = mem - pos_spikes * self.pos_threshold
        mem = mem - neg_spikes * self.neg_threshold

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.mem = self._from_working_dim(mem)

        return spikes


class DLITS(TTLayer):
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
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste

        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)

        pos_mem = pos_mem * self.pos_beta + torch.where(x >= 0, x, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(x <= 0, x, 0.0)

        mem = pos_mem + neg_mem

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        pos_mem = pos_mem - pos_spikes * self.pos_threshold * 0.5
        neg_mem = neg_mem - pos_spikes * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_spikes * self.neg_threshold * 0.5
        neg_mem = neg_mem - neg_spikes * self.neg_threshold * 0.5

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)

        return spikes


class SLITS(TTLayer):
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
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("syn")
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste

        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        syn = self._to_working_dim(self.syn)
        syn = syn * self.alpha + x * (1 - self.alpha)

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + syn

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        mem = mem - pos_spikes * self.pos_threshold
        mem = mem - neg_spikes * self.neg_threshold

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.syn = self._from_working_dim(syn)
        self.mem = self._from_working_dim(mem)

        return spikes


class RLITS(TTLayer):
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
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self._initialize_state("rec")
        self._initialize_state("prev_output")
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste

        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        rec = self._to_working_dim(self.rec)
        prev_output = self._to_working_dim(self.prev_output)
        rec = rec * self.gamma + prev_output * (1 - self.gamma)

        mem_delta = rec * self.rec_weight + x

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + mem_delta

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        mem = mem - pos_spikes * self.pos_threshold
        mem = mem - neg_spikes * self.neg_threshold

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.rec = self._from_working_dim(rec)
        self.mem = self._from_working_dim(mem)
        self.prev_output = spikes

        return spikes


class DSLITS(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_syn")
        self._initialize_state("neg_syn")
        self._register_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self._register_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste

        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_syn = self._to_working_dim(self.pos_syn)
        neg_syn = self._to_working_dim(self.neg_syn)
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = self._from_working_dim(pos_syn)
        self.neg_syn = self._from_working_dim(neg_syn)

        syn = pos_syn + neg_syn

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + torch.where(syn >= 0, syn, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(syn <= 0, syn, 0.0)

        mem = pos_mem + neg_mem

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        pos_mem = pos_mem - pos_spikes * self.pos_threshold * 0.5
        neg_mem = neg_mem - pos_spikes * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_spikes * self.neg_threshold * 0.5
        neg_mem = neg_mem - neg_spikes * self.neg_threshold * 0.5

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)

        return spikes


class DRLITS(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self._initialize_state("pos_rec")
        self._initialize_state("neg_rec")
        self._initialize_state("prev_output")
        self._register_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self._register_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste

        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self._register_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self._register_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_rec = self._to_working_dim(self.pos_rec)
        neg_rec = self._to_working_dim(self.neg_rec)
        prev_output = self._to_working_dim(self.prev_output)

        pos_rec = pos_rec * self.pos_gamma + torch.where(prev_output >= 0, prev_output, 0.0) * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + torch.where(prev_output <= 0, prev_output, 0.0) * (1 - self.neg_gamma)

        self.pos_rec = self._from_working_dim(pos_rec)
        self.neg_rec = self._from_working_dim(neg_rec)

        rec = pos_rec + neg_rec

        pos_mem_delta = torch.where(rec >= 0, rec, 0.0) * self.pos_rec_weight + torch.where(x >= 0, x, 0.0)
        neg_mem_delta = torch.where(rec <= 0, rec, 0.0) * self.neg_rec_weight + torch.where(x <= 0, x, 0.0)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        pos_mem = pos_mem - pos_spikes * self.pos_threshold * 0.5
        neg_mem = neg_mem - pos_spikes * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_spikes * self.neg_threshold * 0.5
        neg_mem = neg_mem - neg_spikes * self.neg_threshold * 0.5

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)
        self.prev_output = spikes

        return spikes


class SRLITS(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("syn")
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self._initialize_state("rec")
        self._initialize_state("prev_output")
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste

        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        syn = self._to_working_dim(x)
        syn = syn * self.alpha + x * (1 - self.alpha)

        rec = self._to_working_dim(self.rec)
        prev_output = self._to_working_dim(self.prev_output)
        rec = rec * self.gamma + prev_output * (1 - self.gamma)
        self.rec = self._from_working_dim(rec)

        mem_delta = rec * self.rec_weight + syn

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + mem_delta

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        mem = mem - pos_spikes * self.pos_threshold
        mem = mem - neg_spikes * self.neg_threshold

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.syn = self._from_working_dim(syn)
        self.mem = self._from_working_dim(mem)
        self.prev_output = spikes

        return spikes


class DSRLITS(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=nn.Sigmoid(),
            deterministic: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_syn")
        self._initialize_state("neg_syn")
        self._register_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self._register_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self._initialize_state("pos_rec")
        self._initialize_state("neg_rec")
        self._initialize_state("prev_output")
        self._register_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self._register_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = self.round_ste if deterministic else self.bernoulli_ste
        self._register_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self._register_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self._register_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self._register_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self._register_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self._register_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_syn = self._to_working_dim(self.pos_syn)
        neg_syn = self._to_working_dim(self.neg_syn)
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = self._from_working_dim(pos_syn)
        self.neg_syn = self._from_working_dim(neg_syn)

        syn = pos_syn + neg_syn

        pos_rec = self._to_working_dim(self.pos_rec)
        neg_rec = self._to_working_dim(self.neg_rec)
        prev_output = self._to_working_dim(self.prev_output)

        pos_rec = pos_rec * self.pos_gamma + torch.where(prev_output >= 0, prev_output, 0.0) * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + torch.where(prev_output <= 0, prev_output, 0.0) * (1 - self.neg_gamma)

        self.pos_rec = self._from_working_dim(pos_rec)
        self.neg_rec = self._from_working_dim(neg_rec)

        rec = pos_rec + neg_rec

        pos_mem_delta = torch.where(rec >= 0, rec, 0.0) * self.pos_rec_weight + torch.where(syn >= 0, syn, 0.0)
        neg_mem_delta = torch.where(rec <= 0, rec, 0.0) * self.neg_rec_weight + torch.where(syn <= 0, syn, 0.0)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        pos_spike_prob = self.spike_fn(mem - self.pos_threshold)
        neg_spike_prob = self.spike_fn(-self.neg_threshold - mem)

        pos_spikes = self.quant_fn(pos_spike_prob)
        neg_spikes = -self.quant_fn(neg_spike_prob)

        pos_mem = pos_mem - pos_spikes * self.pos_threshold * 0.5
        neg_mem = neg_mem - pos_spikes * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_spikes * self.neg_threshold * 0.5
        neg_mem = neg_mem - neg_spikes * self.neg_threshold * 0.5

        pos_spikes = pos_spikes * self.pos_scale
        neg_spikes = neg_spikes * self.neg_scale

        spikes = pos_spikes + neg_spikes

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)
        self.prev_output = spikes

        return spikes
