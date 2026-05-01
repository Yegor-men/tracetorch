import math
import torch
from torch import nn
from ._rnnlayer import Layer as RNNLayer
from typing import Union, Literal


class SimpleRNN(RNNLayer):
    r"""A simple RNN layer, akin to Jordan and Elman networks.
    Uses the input and the previous timestep's output to compute the current timestep's output.

    Args:
        in_features (int): number of input features.
        out_features (int): number of output features. This is the value used as ``num_neurons`` for superclass initialization.
        dim (int, default=-1): the dimension along which the layer operates.

    Attributes:
        H: the hidden state. Stores the previous timestep's output.
        lin: the linear layer used to calculate the output.

    Notes:
        - **Input**: tensor of shape ``[*,in_features,*]`` where ``in_features`` is at index ``dim``.
        - **Output**: tensor of shape ``[*,out_features,*]`` where ``out_features`` is at index ``dim``.

        Concatenates the hidden state ``H`` (previous timestep's output) to the input ``x``, processes both via a linear layer.
        The output is bound by ``tanh``. Records the result into ``H`` and returns it. Pseudocode looks as follows:

        ::

            H = tanh(linear(concatenate(H, x)))
            return H

    Examples::

        # Process 64->10 features along the last dimension
        >>> layer = tt.rnn.SimpleRNN(64, 10)
        >>> input = torch.rand(32, 64)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([32, 10])

        # Process 64->128 features along the color dimension of an image
        >>> layer = tt.rnn.SimpleRNN(64, 128, -3)
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

        self.lin = nn.Linear(in_features + out_features, out_features)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)
        H = self._to_working_dim(self.H)

        H_new = torch.tanh(self.lin(torch.cat([H, x], dim=-1)))

        self.H = self._from_working_dim(H_new)

        return self.H
