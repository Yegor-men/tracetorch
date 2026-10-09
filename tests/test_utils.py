"""Standalone layout, decay conversion, and initialization helpers."""

import math

import pytest
import torch
from torch import nn

import tracetorch as tt


METRICS = ("decay", "halflife", "tau", "horizon")


def _metric_values(decay):
    return {"decay": decay, "halflife": math.log(0.5) / math.log(decay),
            "tau": -1 / math.log(decay), "horizon": 1 / (1 - decay)}


@pytest.mark.parametrize("source", METRICS)
@pytest.mark.parametrize("target", METRICS)
def test_scalar_decay_conversion_has_known_values(source, target):
    values = _metric_values(0.5)
    assert tt.utils.convert_decay(values[source], source, target) == pytest.approx(values[target])


@pytest.mark.parametrize("source", METRICS)
@pytest.mark.parametrize("target", METRICS)
def test_tensor_decay_conversion_has_known_values(source, target, device):
    values = [_metric_values(decay) for decay in (0.5, 0.75)]
    inputs = torch.tensor([value[source] for value in values], dtype=torch.float64, device=device)
    expected = torch.tensor([value[target] for value in values], dtype=torch.float64, device=device)
    actual = tt.utils.convert_decay(inputs, source, target)
    assert actual.dtype == inputs.dtype and actual.shape == inputs.shape
    assert actual.device == inputs.device
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0, 2.0])
def test_decay_conversion_rejects_invalid_base_decay(tensor, value):
    with pytest.raises(ValueError, match="outside"):
        tt.utils.convert_decay(torch.tensor(value) if tensor else value, "decay", "tau")


@pytest.mark.parametrize("source,target", [("unknown", "decay"), ("decay", "unknown")])
def test_decay_conversion_rejects_unknown_metrics(source, target):
    with pytest.raises(ValueError, match="Unknown"):
        tt.utils.convert_decay(0.5, source, target)


@pytest.mark.parametrize("target", ["tau", "halflife"])
def test_tensor_decay_conversion_preserves_autograd(target, device):
    decay = torch.tensor([0.5, 0.75], dtype=torch.float64, device=device, requires_grad=True)
    tt.utils.convert_decay(decay, "decay", target).sum().backward()
    coefficient = math.log(2) if target == "halflife" else 1.0
    expected = coefficient / (decay.detach() * decay.detach().log().square())
    torch.testing.assert_close(decay.grad, expected)


def test_tensor_halflife_to_decay_preserves_autograd(device):
    halflife = torch.tensor([1.0, 4.0], dtype=torch.float64, device=device, requires_grad=True)
    decay = tt.utils.convert_decay(halflife, "halflife", "decay")
    decay.sum().backward()
    expected = decay.detach() * math.log(2) / halflife.detach().square()
    torch.testing.assert_close(halflife.grad, expected)


@pytest.mark.parametrize("inverse,activation,values", [
    (tt.inverse_fn.sigmoid, torch.sigmoid, [0.01, 0.25, 0.5, 0.9, 0.99]),
    (tt.inverse_fn.softplus, nn.functional.softplus, [0.01, 0.5, 1.0, 10.0]),
])
def test_initialization_transform_round_trip(inverse, activation, values):
    values = torch.tensor(values, dtype=torch.float64)
    torch.testing.assert_close(activation(inverse(values)), values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("inverse,value", [
    (tt.inverse_fn.sigmoid, 0.0), (tt.inverse_fn.sigmoid, 1.0),
    (tt.inverse_fn.sigmoid, -0.1), (tt.inverse_fn.sigmoid, 1.1),
    (tt.inverse_fn.softplus, 0.0), (tt.inverse_fn.softplus, -1.0),
])
def test_initialization_transform_rejects_values_outside_domain(inverse, value):
    with pytest.raises(AssertionError):
        inverse(torch.tensor(value))


def test_move_dim_round_trip_and_multiple_axes():
    x = torch.randn(2, 4, 3, 5)
    layout = tt.utils.MoveDim(-3, -1)
    reverse = tt.utils.MoveDim(-1, -3)
    torch.testing.assert_close(reverse(layout(x)), x)
    multi = tt.utils.MoveDim((0, 1), (1, 0))
    torch.testing.assert_close(multi(x), x.movedim((0, 1), (1, 0)))
