import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class SelectiveSSM(RNNLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_features: int,
            decay: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            decay_rank: Literal[0, 1] = 1,
            learn_decay: bool = True,
    ):
        super().__init__(hidden_features, dim)
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.ABD = nn.Linear(in_features, hidden_features * 2 + out_features)
        nn.init.zeros_(self.ABD.bias)
        self.C = nn.Linear(hidden_features, out_features)

        self._register_scale("scale", decay, decay_rank, learn_decay)
        self._initialize_state("mem")

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)

        A, B, D = torch.split(self.ABD(x), [self.hidden_features, self.hidden_features, self.out_features], dim=-1)

        delta = nn.functional.softplus(A)
        decay = torch.exp(delta * -self.scale)

        mem = self._to_working_dim(self.mem)
        mem = mem * decay + B * (1 - decay)

        out = x + nn.functional.silu(D) * self.C(mem)

        out = self._from_working_dim(out)
        self.mem = self._from_working_dim(mem)
        return out
