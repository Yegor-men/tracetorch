import math
import torch
from torch import nn
from ._ssmlayer import Layer as SSMLayer
from typing import Union, Literal
from .. import functional


class SelectiveSSM(SSMLayer):
    def __init__(
            self,
            num_features: int,
            hidden_features: int,
            decay: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            decay_rank: Literal[0, 1] = 1,
            learn_decay: bool = True,
    ):
        super().__init__(hidden_features, dim)
        self.hidden_features = hidden_features
        self.num_features = num_features

        self.ABD = nn.Linear(num_features, hidden_features * 2 + num_features)
        nn.init.zeros_(self.ABD.bias)
        self.C = nn.Linear(hidden_features, num_features)

        self._register_scale("scale", decay, decay_rank, learn_decay)
        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        A, B, D = torch.split(self.ABD(x), [self.hidden_features, self.hidden_features, self.num_features], dim=-1)

        delta = nn.functional.softplus(A)
        decay = torch.exp(delta * -self.scale)

        state = self._to_working_dim(self.state)
        state = state * decay + B * (1 - decay)

        out = x + nn.functional.silu(D) * self.C(state)

        out = self._from_working_dim(out)
        self.state = self._from_working_dim(state)
        return out


class SpikeSSM(SSMLayer):
    def __init__(
            self,
            num_features: int,
            hidden_features: int,
            lif,
            decay: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            decay_rank: Literal[0, 1] = 1,
            learn_decay: bool = True,
    ):
        super().__init__(hidden_features, dim)
        self.hidden_features = hidden_features
        self.num_features = num_features

        self.proj_in = nn.Linear(num_features, num_features)
        self.lif = lif

        self.ABD = nn.Linear(num_features, hidden_features * 2 + num_features)
        nn.init.zeros_(self.ABD.bias)
        self.C = nn.Linear(hidden_features, num_features)
        self._register_scale("scale", decay, decay_rank, learn_decay)
        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        ssm_input = self.lif(self.proj_in(x))

        A, B, D = torch.split(self.ABD(ssm_input), [self.hidden_features, self.hidden_features, self.num_features],
                              dim=-1)
        delta = nn.functional.softplus(A)
        decay = torch.exp(delta * -self.scale)

        state = self._to_working_dim(self.state)
        state = state * decay + B * (1 - decay)

        out = x + nn.functional.silu(D) * self.C(state)

        self.state = self._from_working_dim(state)
        out = self._from_working_dim(out)
        return out


class SelectiveZOHSSM(SSMLayer):
    def __init__(
            self,
            num_features: int,
            hidden_features: int,
            dim: int = -1,
    ):
        super().__init__(hidden_features, dim)
        self.hidden_features = hidden_features
        self.num_features = num_features

        self.ABD = nn.Linear(num_features, hidden_features * 2 + num_features)
        nn.init.zeros_(self.ABD.bias)

        self.C = nn.Linear(hidden_features, num_features)

        self.scale = nn.Parameter(torch.log(torch.arange(1, hidden_features + 1)))
        self._initialize_state("mem")

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        A, B, D = torch.split(self.ABD(x), [self.hidden_features, self.hidden_features, self.num_features], dim=-1)

        delta = nn.functional.softplus(A)
        A_eff = -torch.exp(self.scale)

        delta_A = delta * A_eff
        bar_A = torch.exp(delta_A)
        bar_B = (torch.exp(delta_A) - 1) / (A_eff + 1e-12) * delta

        mem = self._to_working_dim(self.mem)
        mem = mem * bar_A + bar_B * B

        out = x + nn.functional.silu(D) * self.C(mem)

        out = self._from_working_dim(out)
        self.mem = self._from_working_dim(mem)

        return out
