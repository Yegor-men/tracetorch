from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ._tt_infrastructure import TTLayer


class LI(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            ema_beta: bool = False,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)
        self.ema_beta = ema_beta

    def forward(self, x):
        self._ensure_states(x)

        x = self._to_working_dim(x)

        mem = self._to_working_dim(self.mem)
        mem_delta = x * (1 - self.beta) if self.ema_beta else x
        mem = mem * self.beta + mem_delta

        self.mem = self._from_working_dim(mem)

        return self.mem


class DLI(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            ema_pos_beta: bool = False,
            ema_neg_beta: bool = False,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)
        self.ema_pos = ema_pos_beta
        self.ema_neg = ema_neg_beta

    def forward(self, x):
        self._ensure_states(x)

        x = self._to_working_dim(x)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)

        pos_mem_delta = torch.where(x >= 0, x, 0.0)
        neg_mem_delta = torch.where(x <= 0, x, 0.0)

        pos_mem_delta = pos_mem_delta * (1 - self.pos_beta) if self.ema_pos else pos_mem_delta
        neg_mem_delta = neg_mem_delta * (1 - self.neg_beta) if self.ema_neg else neg_mem_delta

        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)

        mem = self.pos_mem + self.neg_mem

        return mem


class SLI(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            ema_alpha: bool = False,
            ema_beta: bool = False,
            learn_alpha: bool = True,
            learn_beta: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("syn")
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)
        self.ema_alpha = ema_alpha

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)
        self.ema_beta = ema_beta

    def forward(self, x):
        self._ensure_states(x)

        x_moved = self._to_working_dim(x)

        syn_moved = self._to_working_dim(self.syn)
        syn_delta = x_moved * (1 - self.alpha) if self.ema_alpha else x_moved
        syn_moved = syn_moved * self.alpha + syn_delta

        mem_moved = self._to_working_dim(self.mem)
        mem_delta = syn_moved * (1 - self.beta) if self.ema_beta else syn_moved
        mem_moved = mem_moved * self.beta + mem_delta

        self.syn = self._from_working_dim(syn_moved)
        self.mem = self._from_working_dim(mem_moved)

        return self.mem


class DSLI(TTLayer):
    def __init__(
            self,
            num_neurons: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            ema_pos_alpha: bool = False,
            ema_neg_alpha: bool = False,
            ema_pos_beta: bool = False,
            ema_neg_beta: bool = False,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_syn")
        self._initialize_state("neg_syn")
        self._register_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self._register_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)
        self.ema_pos_alpha = ema_pos_alpha
        self.ema_neg_alpha = ema_neg_alpha

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)
        self.ema_pos_beta = ema_pos_beta
        self.ema_neg_beta = ema_neg_beta

    def forward(self, x):
        self._ensure_states(x)

        x = self._to_working_dim(x)

        pos_syn = self._to_working_dim(self.pos_syn)
        neg_syn = self._to_working_dim(self.neg_syn)

        pos_syn_delta = torch.where(x >= 0, x, 0.0)
        neg_syn_delta = torch.where(x <= 0, x, 0.0)

        pos_syn_delta = pos_syn_delta * (1 - self.pos_beta) if self.ema_pos_alpha else pos_syn_delta
        neg_syn_delta = neg_syn_delta * (1 - self.neg_beta) if self.ema_neg_beta else neg_syn_delta

        pos_syn = pos_syn * self.pos_beta + pos_syn_delta
        neg_syn = neg_syn * self.neg_beta + neg_syn_delta

        self.pos_syn = self._from_working_dim(pos_syn)
        self.neg_syn = self._from_working_dim(neg_syn)

        syn = self.pos_mem + self.neg_mem

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)

        pos_mem_delta = torch.where(syn >= 0, syn, 0.0)
        neg_mem_delta = torch.where(syn <= 0, syn, 0.0)

        pos_mem_delta = pos_mem_delta * (1 - self.pos_beta) if self.ema_pos else pos_mem_delta
        neg_mem_delta = neg_mem_delta * (1 - self.neg_beta) if self.ema_neg else neg_mem_delta

        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)

        mem = self.pos_mem + self.neg_mem

        return mem
