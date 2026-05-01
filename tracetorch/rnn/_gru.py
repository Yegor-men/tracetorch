import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class GRU(RNNLayer):
    r"""A Gated Recurrent Unit (GRU) layer.
    Uses reset and update gates to control information flow, offering a simpler alternative to LSTM.

    Args:
        in_features (int): number of input features.
        out_features (int): number of output features (automatically becomes the hidden state size). This is the value used as ``num_neurons`` for superclass initialization.
        dim (int, default=-1): the dimension along which the layer operates.

    Attributes:
        H: the hidden state. Stores the previous timestep's output.
        gate_layers: linear layer computing reset and update gates.
        candidate_layer: linear layer computing the candidate hidden state.

    Notes:
        - **Input**: tensor of shape ``[*,in_features,*]`` where ``in_features`` is at index ``dim``.
        - **Output**: tensor of shape ``[*,out_features,*]`` where ``out_features`` is at index ``dim``.

        Computes reset and update gates from concatenated hidden state and input. The reset gate controls
        how much of the previous hidden state to forget, while the update gate balances between old and new
        information. Records the result into ``H`` and returns it. Pseudocode looks as follows:

        ::

            reset_gate, update_gate = chunk(sigmoid(gate_layers(concatenate(H, x))))
            candidate = tanh(candidate_layer(concatenate(H * reset_gate, x)))
            H = H * (1 - update_gate) + update_gate * candidate
            return H

    Examples::

        # Process 64->32 features along the last dimension
        >>> layer = tt.rnn.GRU(64, 32)
        >>> input = torch.rand(16, 64)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])

        # Process 64->128 features along the color dimension of an image
        >>> layer = tt.rnn.GRU(64, 128, -3)
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

        self.gate_layers = nn.Linear(in_features + out_features, 2 * out_features)
        self.candidate_layer = nn.Linear(in_features + out_features, out_features)

    def forward(self, x):
        """Computes the forward pass."""
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
