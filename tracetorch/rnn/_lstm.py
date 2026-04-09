import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class LSTM(RNNLayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dim: int = -1,
    ):
        super().__init__(out_features, dim)

        self._initialize_state("H")
        self._initialize_state("C")

        self.gate_layers = nn.Linear(in_features + out_features, 4 * out_features)

    def forward(self, x):
        self._ensure_states(x)
        x = self._to_working_dim(x)
        H = self._to_working_dim(self.H)
        C = self._to_working_dim(self.C)

        H_x = torch.cat([H, x], dim=-1)

        # 1. Compute all 4 gate inputs at once
        gates = self.gate_layers(H_x)

        # 2. Split into 4 chunks
        # i = input, f = forget, o = output, g = cell candidate (gate)
        i_gate, f_gate, o_gate, g_candidate = gates.chunk(4, dim=-1)

        # 3. Apply activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = torch.tanh(g_candidate)

        # 4. Update the Cell State (C)
        # Forget some old memory + Add some new memory
        C_new = (f * C) + (i * g)

        # 5. Update the Hidden State (H)
        # Filter the long-term memory through the output gate
        H_new = o * torch.tanh(C_new)

        self.C = self._from_working_dim(C_new)
        self.H = self._from_working_dim(H_new)

        return self.H
