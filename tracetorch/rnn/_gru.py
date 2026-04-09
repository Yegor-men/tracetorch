import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class GRU(RNNLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dim: int = -1,
    ):
        super().__init__(out_features, dim)

        self._initialize_state("H")

        self.gate_layers = nn.Linear(in_features + out_features, 2 * out_features)
        self.candidate_layer = nn.Linear(in_features + out_features, out_features)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)
        H = self._to_working_dim(self.H)
        H_x = torch.cat([H, x], dim=-1)

        gates = nn.functional.sigmoid(self.gate_layers(H_x))
        reset_gate, update_gate = gates.chunk(2, dim=-1)

        candidate = torch.tanh(self.candidate_layer(torch.cat([H * reset_gate, x], dim=-1)))

        H = H * (1 - update_gate) + update_gate * candidate

        self.H = self._from_working_dim(H)

        return self.H
