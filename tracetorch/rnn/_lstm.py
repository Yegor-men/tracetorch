import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class LSTM(RNNLayer):
    r"""A Long Short-Term Memory (LSTM) layer.
    Uses input, forget, output gates and a cell state to handle long-term dependencies.

    Args:
        in_features (int): number of input features.
        out_features (int): number of output features (automatically becomes the hidden and cell state size). This is the value used as ``num_neurons`` for superclass initialization.
        dim (int, default=-1): the dimension along which the layer operates.

    Attributes:
        H: the hidden state. Stores the previous timestep's output.
        C: the cell state. Stores long-term memory information.
        gate_layers: linear layer computing all four gates simultaneously.

    Notes:
        - **Input**: tensor of shape ``[*,in_features,*]`` where ``in_features`` is at index ``dim``.
        - **Output**: tensor of shape ``[*,out_features,*]`` where ``out_features`` is at index ``dim``.

        Computes input, forget, output gates and cell candidate from concatenated hidden state and input.
        The forget gate controls what to discard from cell state, input gate controls what new information
        to add, and output gate controls what to expose as the hidden state. Records results into ``H`` and ``C``,
        returns ``H``. Pseudocode looks as follows:

        ::

            i, f, o, g = chunk(sigmoid(gate_layers(concatenate(H, x))), 4)
            C = f * C + i * tanh(g)
            H = o * tanh(C)
            return H

    Examples::

        # Process 64->32 features along the last dimension
        >>> layer = tt.rnn.LSTM(64, 32)
        >>> input = torch.rand(16, 64)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])

        # Process 64->128 features along the color dimension of an image
        >>> layer = tt.rnn.LSTM(64, 128, -3)
        >>> input = torch.rand(32, 64, 28, 28)  # [B, C, H, W] shape
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([32, 128, 28, 28])
    """

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
