"""Numerical contracts for one-timestep recurrent cells."""

from math import exp, tanh

import pytest
import torch

import tracetorch as tt

from tests._cases import RNN_CLASSES
from tests._rnn_reference import pytorch_cell


@pytest.mark.parametrize("layer_class", [tt.rnn.SimpleRNN, tt.rnn.LSTM])
@pytest.mark.parametrize("leading_shape", [(), (2,), (2, 3)])
def test_trajectory_matches_pytorch_cell_with_mapped_weights(layer_class, leading_shape):
    layer = layer_class(3, 2).double()
    reference = pytorch_cell(layer)
    inputs = torch.randn((4,) + leading_shape + (3,), dtype=torch.float64)
    hidden = torch.zeros(inputs[0].reshape(-1, 3).shape[0], 2, dtype=torch.float64)
    cell = torch.zeros_like(hidden)
    with torch.no_grad():
        for x in inputs:
            actual = layer(x)
            if layer_class is tt.rnn.LSTM:
                hidden, cell = reference(x.reshape(-1, 3), (hidden, cell))
                torch.testing.assert_close(layer.C, cell.reshape(leading_shape + (2,)))
            else:
                hidden = reference(x.reshape(-1, 3), hidden)
            torch.testing.assert_close(actual, hidden.reshape(leading_shape + (2,)))
            torch.testing.assert_close(layer.H, actual)


def _scalar_affine(weight, bias, vector):
    return [b + sum(w * x for w, x in zip(row, vector))
            for row, b in zip(weight, bias)]


def test_gru_trajectory_matches_scalar_gate_equations():
    """Check reset-before-projection and traceTorch's update convention."""
    layer = tt.rnn.GRU(3, 2).double()
    gate_weight = layer.gate_layers.weight.detach().tolist()
    gate_bias = layer.gate_layers.bias.detach().tolist()
    candidate_weight = layer.candidate_layer.weight.detach().tolist()
    candidate_bias = layer.candidate_layer.bias.detach().tolist()
    hidden = [0.0, 0.0]
    with torch.no_grad():
        for x in ([0.2, -0.7, 0.9], [-0.4, 0.3, 1.0], [0.0, 0.5, -0.5]):
            gates = [1 / (1 + exp(-value)) for value in
                     _scalar_affine(gate_weight, gate_bias, hidden + x)]
            reset, update = gates[:2], gates[2:]
            reset_hidden = [h * r for h, r in zip(hidden, reset)]
            candidate = [tanh(value) for value in
                         _scalar_affine(candidate_weight, candidate_bias, reset_hidden + x)]
            hidden = [(1 - z) * h + z * c for h, z, c in zip(hidden, update, candidate)]
            torch.testing.assert_close(layer(torch.tensor(x, dtype=torch.float64)),
                                       torch.tensor(hidden, dtype=torch.float64),
                                       rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("layer_class", RNN_CLASSES)
def test_parallel_sequences_match_separate_cells(layer_class):
    batched = layer_class(3, 2)
    instances = [layer_class(3, 2) for _ in range(2)]
    for instance in instances:
        instance.load_state_dict(batched.state_dict())
    with torch.no_grad():
        for x in torch.randn(4, 2, 3):
            expected = torch.stack([instance(row) for instance, row in zip(instances, x)])
            torch.testing.assert_close(batched(x), expected)
            for name in batched._state_names:
                expected_state = torch.stack([getattr(instance, name) for instance in instances])
                torch.testing.assert_close(getattr(batched, name), expected_state)


@pytest.mark.parametrize("layer_class", [tt.rnn.SimpleRNN, tt.rnn.GRU, tt.rnn.LSTM])
@pytest.mark.parametrize("shape", [(4,), (2, 4), (2, 3, 4)])
def test_rnn_states_use_output_shape_and_features_last(layer_class, shape):
    layer = layer_class(4, 5)
    x = torch.randn(shape)
    layer(x)
    second = layer(x)
    assert second.shape == shape[:-1] + (5,)
    assert all(getattr(layer, name).shape == second.shape for name in layer._state_names)
    layer.detach_states()
    assert all(not getattr(layer, name).requires_grad for name in layer._state_names)
    layer.reset_states()
    assert layer(torch.randn(7, 4)).shape == (7, 5)
