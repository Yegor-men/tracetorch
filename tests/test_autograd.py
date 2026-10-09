"""Analytical, numerical, and surrogate-gradient contracts."""

import pytest
import torch

import tracetorch as tt

from tests._cases import SNN_CLASSES
from tests._rnn_reference import pytorch_cell


@pytest.mark.parametrize("detach", [False, True], ids=["full_history", "detached_history"])
def test_li_input_and_raw_decay_gradients_are_analytical(detach):
    layer = tt.snn.LI(1, beta=torch.tensor(0.5, dtype=torch.float64))
    first = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    layer(first)
    if detach:
        layer.detach_states()
    result = layer(second)
    torch.testing.assert_close(result, torch.tensor([4.0], dtype=torch.float64))
    result.sum().backward()
    if detach:
        assert first.grad is None
    else:
        torch.testing.assert_close(first.grad, torch.tensor([0.5], dtype=torch.float64))
    torch.testing.assert_close(second.grad, torch.ones_like(second))
    # d(mem2)/d(raw_beta) = input1 * beta * (1 - beta).
    torch.testing.assert_close(layer.raw_beta.grad, torch.tensor(0.5, dtype=torch.float64))


def test_li_three_step_gradient_includes_all_temporal_paths():
    layer = tt.snn.LI(1, beta=torch.tensor(0.5, dtype=torch.float64))
    sequence = torch.tensor([[2.0], [3.0], [-1.0]], dtype=torch.float64, requires_grad=True)
    for x in sequence:
        result = layer(x)
    result.sum().backward()
    torch.testing.assert_close(sequence.grad, torch.tensor([[0.25], [0.5], [1.0]],
                                                          dtype=torch.float64))
    # mem3 = beta**2 * input1 + beta * input2 + input3.
    torch.testing.assert_close(layer.raw_beta.grad, torch.tensor(1.25, dtype=torch.float64))


def test_reset_discards_values_and_temporal_gradient_paths():
    layer = tt.snn.LI(1, beta=0.5)
    first, second = (torch.tensor([value], requires_grad=True) for value in (2.0, 3.0))
    layer(first)
    layer.reset_states()
    result = layer(second)
    result.sum().backward()
    torch.testing.assert_close(result, second)
    assert first.grad is None
    torch.testing.assert_close(second.grad, torch.ones_like(second))
    torch.testing.assert_close(layer.raw_beta.grad, torch.zeros_like(layer.raw_beta))


@pytest.mark.parametrize("learnable", [False, True])
def test_frozen_parameters_do_not_block_input_gradients(learnable):
    layer = tt.snn.LI(1, beta=0.5, learn_beta=learnable)
    x = torch.tensor([2.0], requires_grad=True)
    layer(x)
    layer(torch.zeros_like(x)).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([0.5]))
    if learnable:
        torch.testing.assert_close(layer.raw_beta.grad, torch.tensor([0.5]))
    else:
        assert layer.raw_beta.grad is None


@pytest.mark.parametrize("spike_fn", [tt.snn.spike_fn.smooth, tt.snn.spike_fn.deterministic,
                                      tt.snn.spike_fn.stochastic], ids=lambda fn: fn.__name__)
def test_spike_backward_matches_prescribed_surrogate(spike_fn):
    x = torch.tensor([-1.0, -0.25, 0.0, 0.5], dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([-2.0, 0.5, 1.5, 3.0], dtype=torch.float64)
    spike_fn(x).backward(upstream)
    # d sigmoid(4x) / dx = sech(2x)**2; independent expression for the slope.
    torch.testing.assert_close(x.grad, upstream / torch.cosh(2 * x.detach()).square(),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("layer_class", SNN_CLASSES, ids=lambda cls: cls.__name__)
def test_smooth_temporal_dynamics_pass_numerical_gradcheck(layer_class):
    name = layer_class.__name__
    if name.startswith("D"):
        parameters = {"pos_beta": torch.tensor([0.4, 0.7], dtype=torch.float64),
                      "neg_beta": torch.tensor([0.3, 0.6], dtype=torch.float64)}
    else:
        parameters = {"beta": torch.tensor([0.4, 0.7], dtype=torch.float64)}
    if name.endswith("LIB"):
        parameters.update(threshold=torch.tensor([0.8, 1.1], dtype=torch.float64),
                          spike_fn=tt.snn.spike_fn.smooth)
    elif name.endswith(("LIT", "LITS")):
        parameters.update(pos_threshold=torch.tensor([0.8, 1.1], dtype=torch.float64),
                          neg_threshold=torch.tensor([0.9, 0.7], dtype=torch.float64),
                          spike_fn=tt.snn.spike_fn.smooth)
    layer = layer_class(2, **parameters).double()
    sequence = torch.tensor([[0.2, -0.3], [0.4, 0.1], [-0.1, 0.5]],
                            dtype=torch.float64, requires_grad=True)
    names = tuple(name for name, _ in layer.named_parameters())
    raw_values = tuple(value.detach().clone().requires_grad_() for value in layer.parameters())

    def trajectory(inputs, *raw):
        # Numerical checks call repeatedly; every call must start at the same state.
        layer.reset_states()
        for x in inputs:
            output = torch.func.functional_call(layer, dict(zip(names, raw)), (x,))
        return output

    assert torch.autograd.gradcheck(trajectory, (sequence,) + raw_values)


@pytest.mark.parametrize("layer_class", [cls for cls in SNN_CLASSES
                                       if cls.__name__.endswith(("LIT", "LITS"))],
                         ids=lambda cls: cls.__name__)
def test_ternary_reset_has_gradients_for_both_thresholds(layer_class, device):
    layer = layer_class(1, pos_threshold=0.7, neg_threshold=1.1,
                        spike_fn=tt.snn.spike_fn.smooth).to(device=device, dtype=torch.float64)
    x = torch.zeros(1, dtype=torch.float64, device=device)
    layer.zero_states(x)
    # Both branches are present even though their sum is positive.
    layer.pos_prev_output = x.new_tensor([0.3])
    layer.neg_prev_output = x.new_tensor([-0.2])
    layer(x)
    if hasattr(layer, "pos_mem"):
        membrane = layer.pos_mem + layer.neg_mem
        reset_decay = (layer.pos_beta.detach() + layer.neg_beta.detach()) / 2
    else:
        membrane = layer.mem
        reset_decay = layer.beta.detach()
    membrane.sum().backward()
    expected_positive = -0.3 * reset_decay * torch.sigmoid(layer.raw_pos_threshold.detach())
    expected_negative = 0.2 * reset_decay * torch.sigmoid(layer.raw_neg_threshold.detach())
    torch.testing.assert_close(layer.raw_pos_threshold.grad, expected_positive)
    torch.testing.assert_close(layer.raw_neg_threshold.grad, expected_negative)


@pytest.mark.parametrize("layer_class", [tt.snn.DSLI, tt.snn.DSLIEMA, tt.snn.DSLIB,
                                       tt.snn.DSRLIB, tt.snn.DSLIT, tt.snn.DSRLIT,
                                       tt.snn.DSLITS, tt.snn.DSRLITS])
def test_dual_membranes_receive_both_synaptic_histories_and_gradients(layer_class, device):
    layer = layer_class(1, pos_alpha=0.5, neg_alpha=0.25).to(device=device, dtype=torch.float64)
    x = torch.zeros(1, dtype=torch.float64, device=device)
    layer.zero_states(x)
    positive = x.new_tensor([2.0], requires_grad=True)
    negative = x.new_tensor([-1.0], requires_grad=True)
    layer.pos_syn, layer.neg_syn = positive, negative
    layer(x)
    pos_gain = 1 - layer.pos_beta.detach() if layer_class is tt.snn.DSLIEMA else x.new_ones(1)
    neg_gain = 1 - layer.neg_beta.detach() if layer_class is tt.snn.DSLIEMA else x.new_ones(1)
    torch.testing.assert_close(layer.pos_mem, pos_gain)
    torch.testing.assert_close(layer.neg_mem, -0.25 * neg_gain)
    (layer.pos_mem + layer.neg_mem).sum().backward()
    torch.testing.assert_close(positive.grad, 0.5 * pos_gain)
    torch.testing.assert_close(negative.grad, 0.25 * neg_gain)
    torch.testing.assert_close(layer.raw_pos_alpha.grad, 0.5 * pos_gain)
    torch.testing.assert_close(layer.raw_neg_alpha.grad, -0.1875 * neg_gain)


@pytest.mark.parametrize("layer_class", [tt.snn.DRLIB, tt.snn.DSRLIB, tt.snn.DRLIT,
                                       tt.snn.DSRLIT, tt.snn.DRLITS, tt.snn.DSRLITS])
def test_dual_membranes_receive_both_recurrent_histories_and_gradients(layer_class, device):
    layer = layer_class(1, pos_gamma=0.5, neg_gamma=0.25,
                        pos_rec_weight=2.0, neg_rec_weight=3.0).to(device=device, dtype=torch.float64)
    x = torch.zeros(1, dtype=torch.float64, device=device)
    layer.zero_states(x)
    positive = x.new_tensor([2.0], requires_grad=True)
    negative = x.new_tensor([-1.0], requires_grad=True)
    layer.pos_rec, layer.neg_rec = positive, negative
    layer(x)
    torch.testing.assert_close(layer.pos_mem, x.new_tensor([2.0]))
    torch.testing.assert_close(layer.neg_mem, x.new_tensor([-0.75]))
    (layer.pos_mem + layer.neg_mem).sum().backward()
    torch.testing.assert_close(positive.grad, x.new_tensor([1.0]))
    torch.testing.assert_close(negative.grad, x.new_tensor([0.75]))
    torch.testing.assert_close(layer.raw_pos_rec_weight.grad, x.new_tensor([1.0]))
    torch.testing.assert_close(layer.raw_neg_rec_weight.grad, x.new_tensor([-0.25]))
    torch.testing.assert_close(layer.raw_pos_gamma.grad, x.new_tensor([1.0]))
    torch.testing.assert_close(layer.raw_neg_gamma.grad, x.new_tensor([-0.5625]))


@pytest.mark.parametrize("layer_class", [tt.rnn.SimpleRNN, tt.rnn.LSTM])
def test_rnn_input_and_weight_gradients_match_independent_pytorch_cells(layer_class):
    layer = layer_class(3, 2).double()
    reference = pytorch_cell(layer)
    inputs = torch.randn(3, 2, 3, dtype=torch.float64, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    hidden = torch.zeros(2, 2, dtype=torch.float64)
    cell = torch.zeros_like(hidden)
    for x in inputs:
        actual = layer(x)
    for x in reference_inputs:
        if layer_class is tt.rnn.LSTM:
            hidden, cell = reference(x, (hidden, cell))
        else:
            hidden = reference(x, hidden)
    actual.square().sum().backward()
    hidden.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    affine = layer.lin if layer_class is tt.rnn.SimpleRNN else layer.gate_layers
    expected_weight = torch.cat((reference.weight_hh.grad, reference.weight_ih.grad), dim=1)
    expected_bias = reference.bias_ih.grad
    if layer_class is tt.rnn.LSTM:
        expected_weight = torch.cat([expected_weight.chunk(4)[i] for i in (0, 1, 3, 2)])
        expected_bias = torch.cat([expected_bias.chunk(4)[i] for i in (0, 1, 3, 2)])
    torch.testing.assert_close(affine.weight.grad, expected_weight)
    torch.testing.assert_close(affine.bias.grad, expected_bias)


def test_gru_gradients_match_independent_projection_equations():
    layer = tt.rnn.GRU(3, 2).double()
    inputs = torch.randn(3, 2, 3, dtype=torch.float64, requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weights = [param.detach().clone().requires_grad_() for param in layer.parameters()]
    gate_weight, gate_bias, candidate_weight, candidate_bias = weights
    hidden = torch.zeros(2, 2, dtype=torch.float64)
    for x in inputs:
        actual = layer(x)
    for x in ref_inputs:
        # Separate input/hidden projections avoid using the production concatenation.
        gates = hidden @ gate_weight[:, :2].T + x @ gate_weight[:, 2:].T + gate_bias
        reset = torch.sigmoid(gates[:, :2])
        update = torch.sigmoid(gates[:, 2:])
        candidate = torch.tanh((hidden * reset) @ candidate_weight[:, :2].T
                               + x @ candidate_weight[:, 2:].T + candidate_bias)
        hidden = (1 - update) * hidden + update * candidate
    actual.square().sum().backward()
    hidden.square().sum().backward()
    torch.testing.assert_close(inputs.grad, ref_inputs.grad)
    for parameter, reference in zip(layer.parameters(), weights):
        torch.testing.assert_close(parameter.grad, reference.grad)


@pytest.mark.parametrize("layer_class", SNN_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("shape", [(4,), (2, 4), (2, 3, 4)])
def test_snn_layers_support_backward_for_leading_instance_dimensions(layer_class, shape):
    layer = layer_class(4)
    inputs = [torch.randn(shape, requires_grad=True) for _ in range(3)]
    baseline = [layer(x) for x in inputs]
    assert all(output.shape == shape for output in baseline)
    assert all(getattr(layer, name).shape == shape for name in layer._state_names)
    sum(output.sum() for output in baseline).backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs)
    assert all(param.grad is not None and torch.isfinite(param.grad).all()
               for param in layer.parameters())
