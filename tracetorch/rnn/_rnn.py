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
        out_features (int): number of output features.

    Attributes:
        H: the hidden state. Stores the previous timestep's output.
        lin: the linear layer used to calculate the output.

    Notes:
        - **Input**: tensor of shape ``[..., in_features]``.
        - **Output**: tensor of shape ``[..., out_features]``.

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

        # Move image channels to the last dimension explicitly
        >>> layer = tt.rnn.SimpleRNN(64, 128)
        >>> input = torch.rand(32, 64, 28, 28)
        >>> output = layer(input.movedim(-3, -1)).movedim(-1, -3)
        >>> print(output.shape)
        torch.Size([32, 128, 28, 28])
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        super().__init__()

        self.define_state("H", (out_features,))

        self.lin = nn.Linear(in_features + out_features, out_features)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        H = self.H

        H_new = torch.tanh(self.lin(torch.cat([H, x], dim=-1)))

        self.H = H_new

        return self.H
