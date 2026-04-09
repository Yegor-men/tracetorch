import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class SimpleRNN(RNNLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dim: int = -1,
    ):
        super().__init__(out_features, dim)

        self._initialize_state("H")

        self.lin = nn.Linear(in_features + out_features, out_features)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)
        H = self._to_working_dim(self.H)

        H_new = torch.tanh(self.lin(torch.cat([H, x], dim=-1)))

        self.H = self._from_working_dim(H_new)

        return self.H
