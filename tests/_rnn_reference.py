"""Weight mappings to independent PyTorch cells where equations coincide."""

import torch
from torch import nn


def pytorch_cell(layer):
    """Map hidden-first concatenation and LSTM's i,f,o,g gate ordering."""
    if type(layer).__name__ == "SimpleRNN":
        affine = layer.lin
        hidden_size = affine.out_features
        cell = nn.RNNCell(affine.in_features - hidden_size, hidden_size,
                          nonlinearity="tanh").to(affine.weight)
        weight, bias = affine.weight, affine.bias
    elif type(layer).__name__ == "LSTM":
        affine = layer.gate_layers
        hidden_size = affine.out_features // 4
        cell = nn.LSTMCell(affine.in_features - hidden_size, hidden_size).to(affine.weight)
        # traceTorch: i,f,o,g; PyTorch: i,f,g,o.
        weight = torch.cat([affine.weight.chunk(4)[i] for i in (0, 1, 3, 2)])
        bias = torch.cat([affine.bias.chunk(4)[i] for i in (0, 1, 3, 2)])
    else:
        raise ValueError("traceTorch GRU resets before the candidate projection; "
                         "PyTorch GRUCell is not an equivalent reference")
    with torch.no_grad():
        cell.weight_hh.copy_(weight[:, :hidden_size])
        cell.weight_ih.copy_(weight[:, hidden_size:])
        cell.bias_ih.copy_(bias)
        cell.bias_hh.zero_()
    return cell
