"""SNN dynamics, firing decisions, and per-neuron parameter construction."""

import pytest
import torch
from torch import nn

import tracetorch as tt

from tests._cases import SNN_NAMES
from tests._references import ScalarNeuron


def _neuron_configuration(name):
    """Decode the public naming scheme, not implementation attributes."""
    suffix = next(family for family in ("LIEMA", "LITS", "LIB", "LIT", "LI")
                  if name.endswith(family))
    prefix = name[:-len(suffix)]
    return {
        "family": "LI" if suffix == "LIEMA" else suffix,
        "dual": "D" in prefix,
        "synaptic": "S" in prefix,
        "recurrent": "R" in prefix,
        "ema": suffix == "LIEMA",
    }


def _neuron_parameters(configuration):
    values = {
        "beta": 0.5, "alpha": 0.25, "gamma": 0.75, "rec_weight": 0.4,
        "pos_beta": 0.4, "neg_beta": 0.8,
        "pos_alpha": 0.3, "neg_alpha": 0.7,
        "pos_gamma": 0.2, "neg_gamma": 0.6,
        "pos_rec_weight": 0.6, "neg_rec_weight": 0.9,
        "threshold": 0.7, "pos_threshold": 0.8, "neg_threshold": 0.6,
        "pos_scale": 2.0, "neg_scale": 3.0,
    }
    names = []
    for branch in (("pos_", "neg_") if configuration["dual"] else ("",)):
        names.append(branch + "beta")
        if configuration["synaptic"]:
            names.append(branch + "alpha")
        if configuration["recurrent"]:
            names.extend((branch + "gamma", branch + "rec_weight"))
    if configuration["family"] == "LIB":
        names.append("threshold")
    elif configuration["family"] in ("LIT", "LITS"):
        names.extend(("pos_threshold", "neg_threshold"))
    if configuration["family"] == "LITS":
        names.extend(("pos_scale", "neg_scale"))
    return {name: values[name] for name in names}


def test_public_snn_inventory():
    assert set(tt.snn.__all__) - {"Layer", "spike_fn"} == set(SNN_NAMES)


def test_deterministic_spikes_include_the_threshold():
    x = torch.tensor([-1.0, -0.001, 0.0, 0.001, 1.0])
    torch.testing.assert_close(tt.snn.spike_fn.deterministic(x), torch.tensor([0., 0., 1., 1., 1.]))


def test_smooth_spikes_have_known_values():
    x = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64)
    expected = 1 / (1 + torch.exp(-4 * x))
    torch.testing.assert_close(tt.snn.spike_fn.smooth(x), expected, rtol=1e-12, atol=1e-12)


def test_stochastic_spikes_are_binary_with_correct_extremes_and_balanced_threshold():
    torch.testing.assert_close(tt.snn.spike_fn.stochastic(torch.tensor([-100.0, 100.0])),
                               torch.tensor([0.0, 1.0]))
    events = tt.snn.spike_fn.stochastic(torch.zeros(20000))
    assert ((events == 0) | (events == 1)).all()
    # A wide deterministic-seed tolerance, not an exact RNG sequence contract.
    assert abs(events.mean().item() - 0.5) < 0.025


@pytest.mark.parametrize("name,smooth", [
    (name, smooth) for name in SNN_NAMES
    for smooth in ([False] if _neuron_configuration(name)["family"] == "LI" else [False, True])
], ids=[
    name + ("-smooth" if smooth else "-hard") for name in SNN_NAMES
    for smooth in ([False] if _neuron_configuration(name)["family"] == "LI" else [False, True])
])
def test_all_snn_trajectories_match_independent_scalar_neurons(name, smooth):
    configuration = _neuron_configuration(name)
    first = _neuron_parameters(configuration)
    second = {
        key: (1 - value if key.endswith(("alpha", "beta", "gamma")) else
              -0.5 * value if key.endswith("rec_weight") else 1.25 * value)
        for key, value in first.items()
    }
    parameters = {key: torch.tensor([first[key], second[key]], dtype=torch.float64)
                  for key in first}
    spike_options = {"spike_fn": tt.snn.spike_fn.smooth} if smooth else {}
    layer = getattr(tt.snn, name)(2, **parameters, **spike_options)
    references = [ScalarNeuron(parameters=p, smooth=smooth, **configuration) for p in (first, second)]

    with torch.no_grad():
        for value in (0.0, 3.0, -4.0, 0.125, 2.0, 0.0, -2.0, -0.25):
            inputs = (value, -0.75 * value)
            expected = torch.tensor([ref.step(x) for ref, x in zip(references, inputs)],
                                    dtype=torch.float64)
            torch.testing.assert_close(layer(torch.tensor(inputs, dtype=torch.float64)), expected)
            assert layer._state_names == set(references[0].states)
            for state_name in references[0].states:
                expected_state = torch.tensor([ref.states[state_name] for ref in references],
                                              dtype=torch.float64)
                torch.testing.assert_close(getattr(layer, state_name), expected_state,
                                           rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("layer_class,expected", [
    (tt.snn.LI, [2.0, 4.0, 1.0]),
    (tt.snn.LIEMA, [1.0, 2.0, 0.5]),
])
def test_integrator_hand_calculated_trajectory(layer_class, expected):
    layer = layer_class(1, beta=torch.tensor(0.5, dtype=torch.float64))
    for x, membrane in zip((2.0, 3.0, -1.0), expected):
        actual = layer(torch.tensor([x], dtype=torch.float64))
        torch.testing.assert_close(actual, torch.tensor([membrane], dtype=torch.float64))


@pytest.mark.parametrize("layer_class,expected", [
    (tt.snn.LIB, [0.0, 0.0, 0.0, 1.0, 1.0]),
    (tt.snn.LIT, [-1.0, -1.0, 0.0, 1.0, 1.0]),
    (tt.snn.LITS, [-3.0, -3.0, 0.0, 2.0, 2.0]),
])
def test_threshold_boundaries_and_signed_output(layer_class, expected):
    parameters = {"threshold": torch.tensor(1.0, dtype=torch.float64)}
    if layer_class is not tt.snn.LIB:
        parameters = {"pos_threshold": parameters["threshold"],
                      "neg_threshold": parameters["threshold"]}
    if layer_class is tt.snn.LITS:
        parameters.update(pos_scale=2.0, neg_scale=3.0)
    layer = layer_class(5, **parameters)
    x = torch.tensor([-1.01, -1.0, 0.0, 1.0, 1.01], dtype=torch.float64)
    torch.testing.assert_close(layer(x), torch.tensor(expected, dtype=torch.float64))


def test_binary_reset_is_delayed_until_next_step_and_precedes_decay():
    layer = tt.snn.LIB(1, beta=0.5, threshold=1.0)
    torch.testing.assert_close(layer(torch.tensor([2.0])), torch.ones(1))
    # Emitting a spike does not immediately erase the stored membrane.
    torch.testing.assert_close(layer.mem, torch.tensor([2.0]))
    torch.testing.assert_close(layer(torch.zeros(1)), torch.zeros(1))
    # (2 - 1) * 0.5, not 2 * 0.5 - 1.
    torch.testing.assert_close(layer.mem, torch.tensor([0.5]))


def test_scaled_events_do_not_scale_reset_or_recurrent_history():
    layer = tt.snn.RLITS(1, beta=0.5, gamma=0.5, rec_weight=0.25,
                         pos_scale=4.0, neg_scale=3.0)
    torch.testing.assert_close(layer(torch.tensor([2.0])), torch.tensor([4.0]))
    torch.testing.assert_close(layer.pos_prev_output, torch.ones(1))
    torch.testing.assert_close(layer.neg_prev_output, torch.zeros(1))
    layer(torch.zeros(1))
    torch.testing.assert_close(layer.rec, torch.tensor([0.5]))
    torch.testing.assert_close(layer.mem, torch.tensor([0.625]))


@pytest.mark.parametrize("layer_class", [tt.snn.LIT, tt.snn.LITS])
def test_simultaneous_ternary_events_do_not_cancel_reset(layer_class):
    # Stochastic firing can emit both events; make that case deterministic here.
    layer = layer_class(1, beta=0.5, pos_threshold=0.5, neg_threshold=1.5,
                        spike_fn=torch.ones_like)
    layer(torch.zeros(1))
    torch.testing.assert_close(layer.pos_prev_output, torch.ones(1))
    torch.testing.assert_close(layer.neg_prev_output, -torch.ones(1))
    layer(torch.zeros(1))
    # The net output was zero, but the reset is 1 * 0.5 - 1 * 1.5 = -1.
    torch.testing.assert_close(layer.mem, torch.tensor([0.5]))


@pytest.mark.parametrize("name", [name for name in SNN_NAMES if name.endswith("LITS")])
@pytest.mark.parametrize("scales", [(2.0, 3.0), (-2.0, 3.0), (0.0, 0.0)])
def test_output_scales_never_change_internal_ternary_dynamics(name, scales, device):
    scaled = getattr(tt.snn, name)(1, pos_scale=scales[0], neg_scale=scales[1],
                                   spike_fn=tt.snn.spike_fn.smooth).to(device=device, dtype=torch.float64)
    unscaled = getattr(tt.snn, name[:-1])(1, spike_fn=tt.snn.spike_fn.smooth).to(
        device=device, dtype=torch.float64)
    weights = {key: value for key, value in scaled.state_dict().items()
               if key not in ("raw_pos_scale", "raw_neg_scale")}
    unscaled.load_state_dict(weights)
    with torch.no_grad():
        for value in (1.2, -0.7, 0.1, -1.5, 0.3):
            x = torch.tensor([value], dtype=torch.float64, device=device)
            output = scaled(x)
            unscaled(x)
            expected = scales[0] * scaled.pos_prev_output + scales[1] * scaled.neg_prev_output
            torch.testing.assert_close(output, expected)
            assert scaled._state_names == unscaled._state_names
            for state_name in scaled._state_names:
                torch.testing.assert_close(getattr(scaled, state_name), getattr(unscaled, state_name))


@pytest.mark.parametrize("layer_class", [tt.snn.SLIB, tt.snn.SRLIB,
                                         tt.snn.SRLIT, tt.snn.SRLITS])
def test_synaptic_trace_accumulates_previous_state(layer_class):
    layer = layer_class(4, alpha=0.5, learn_alpha=False)
    layer(torch.ones(2, 4))
    torch.testing.assert_close(layer.syn, torch.full((2, 4), 0.5))
    layer(torch.ones(2, 4))
    torch.testing.assert_close(layer.syn, torch.full((2, 4), 0.75))


@pytest.mark.parametrize("layer_class", [tt.snn.LI, tt.snn.LIB, tt.snn.DSRLITS])
def test_parallel_instances_have_independent_state(layer_class):
    batched = layer_class(2)
    first, second = layer_class(2), layer_class(2)
    first.load_state_dict(batched.state_dict())
    second.load_state_dict(batched.state_dict())
    with torch.no_grad():
        for x in (torch.tensor([[2.0, -2.0], [0.0, 0.5]]),
                  torch.tensor([[0.0, 0.0], [-3.0, 2.0]])):
            output = batched(x)
            torch.testing.assert_close(output[0], first(x[0]))
            torch.testing.assert_close(output[1], second(x[1]))
            for name in batched._state_names:
                torch.testing.assert_close(getattr(batched, name)[0], getattr(first, name))
                torch.testing.assert_close(getattr(batched, name)[1], getattr(second, name))


def test_scalar_and_per_neuron_parameter_initialization():
    layer = tt.snn.RLITS(4, beta_rank=0, pos_scale_rank=0, rec_weight_rank=0,
                        learn_beta=False, learn_pos_scale=False, learn_rec_weight=False)
    for name in ("beta", "pos_scale", "rec_weight"):
        assert getattr(layer, f"raw_{name}").shape == ()
        assert isinstance(getattr(layer, f"raw_{name}"), nn.Parameter)
    assert layer.raw_neg_scale.shape == (4,)
    tensor_layer = tt.snn.LI(4, beta=torch.tensor(0.8))
    assert tensor_layer.raw_beta.shape == ()
    vector_layer = tt.snn.LI(4, beta=torch.full((4,), 0.8), beta_rank=0)
    assert vector_layer.raw_beta.shape == (4,)
    with pytest.raises(ValueError):
        tt.snn.LI(4, beta=torch.ones(3) * 0.5)
    with pytest.raises(ValueError):
        tt.snn.LI(4, beta_rank=2)
