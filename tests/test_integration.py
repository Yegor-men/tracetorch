"""Composition, inference compilation, training, and checkpoint workflows."""

import pytest
import torch
from torch import nn

import tracetorch as tt
from tracetorch import snn

from tests._cases import SNN_CLASSES


class MixedModel(tt.Model):
    """Small differentiable SNN/RNN composition, with no dataset dependency."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 4),
            tt.snn.LIB(4, beta=0.7, threshold=0.3, spike_fn=tt.snn.spike_fn.smooth),
            tt.rnn.GRU(4, 4),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def _training_step(model, optimizer, output, target):
    loss = nn.functional.mse_loss(output, target)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


@pytest.mark.parametrize("layer_class", [tt.snn.LIT, tt.snn.LITS,
                                       tt.snn.DSRLIT, tt.snn.DSRLITS])
def test_ternary_state_checkpoint_preserves_both_event_branches(layer_class, device):
    model, restored = tt.Model(), tt.Model()
    model.neuron = layer_class(2, spike_fn=tt.snn.spike_fn.smooth).to(device)
    restored.neuron = layer_class(2, spike_fn=tt.snn.spike_fn.smooth).to(device)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        for x in (torch.tensor([0.2, -0.1], device=device),
                  torch.tensor([-0.3, 0.4], device=device)):
            model.neuron(x)
        states = model.save_states()
        assert "neuron.pos_prev_output" in states and "neuron.neg_prev_output" in states
        assert "neuron.prev_output" not in states
        restored.load_states(states)
        x = torch.tensor([0.1, -0.2], device=device)
        torch.testing.assert_close(model.neuron(x), restored.neuron(x))
        for state_name in model.neuron._state_names:
            torch.testing.assert_close(getattr(model.neuron, state_name),
                                       getattr(restored.neuron, state_name))


@pytest.mark.parametrize("online", [False, True], ids=["bptt", "online"])
def test_mixed_model_trains_and_resets_between_variable_batches(device, online):
    model = MixedModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    for batch_size in (2, 1, 3):
        model.reset_states()
        sequence = torch.randn(4, batch_size, 3, device=device)
        targets = torch.randn(4, batch_size, 2, device=device)
        losses = []
        for x, target in zip(sequence, targets):
            output = model(x)
            assert output.shape == (batch_size, 2)
            if online:
                _training_step(model, optimizer, output, target)
                model.detach_states()
                assert not model.net[1].mem.requires_grad
                assert not model.net[2].H.requires_grad
            else:
                losses.append(nn.functional.mse_loss(output, target))
        if not online:
            loss = torch.stack(losses).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        assert model.net[1].mem.shape == model.net[2].H.shape == (batch_size, 4)
        assert ((model.net[1].beta > 0) & (model.net[1].beta < 1)).all()
        assert (model.net[1].threshold > 0).all()

    changes = {name: not torch.equal(value, initial[name]) for name, value in model.named_parameters()}
    assert changes["net.1.raw_beta"] and changes["net.1.raw_threshold"]
    assert changes["net.0.weight"] and changes["net.2.gate_layers.weight"]


def test_checkpoint_resumes_inference_and_an_optimizer_step(tmp_path, device):
    model = MixedModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for x, target in zip(torch.randn(3, 2, 3, device=device),
                         torch.randn(3, 2, 2, device=device)):
        _training_step(model, optimizer, model(x), target)
        model.detach_states()

    # Persist weights, runtime state, and optimizer history as separate files.
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    torch.save(model.save_states(), tmp_path / "states.pt")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    restored = MixedModel().to(device)
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", map_location=device, weights_only=True))
    restored.load_states(torch.load(tmp_path / "states.pt", map_location=device, weights_only=True))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(torch.load(tmp_path / "optimizer.pt", map_location=device,
                                                  weights_only=True))

    x, target = torch.randn(2, 3, device=device), torch.randn(2, 2, device=device)
    actual, resumed = model(x), restored(x)
    torch.testing.assert_close(actual, resumed)
    _training_step(model, optimizer, actual, target)
    _training_step(restored, restored_optimizer, resumed, target)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, restored.state_dict()[name])
    model.detach_states()
    restored.detach_states()
    model.eval()
    restored.eval()
    restored.compile_parameters()
    with torch.no_grad():
        for x in torch.randn(3, 2, 3, device=device):
            torch.testing.assert_close(model(x), restored(x))


def test_channel_first_composition_keeps_states_feature_last():
    network = nn.Sequential(tt.utils.MoveDim(-3, -1), tt.rnn.GRU(4, 6),
                            tt.utils.MoveDim(-1, -3))
    assert network(torch.randn(2, 4, 3, 5)).shape == (2, 6, 3, 5)
    assert network[1].H.shape == (2, 3, 5, 6)


class CNNModel(tt.Model):
    def __init__(self, c, n_labels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(c, 16, 3),
            tt.utils.MoveDim(-3, -1),
            snn.LIB(16, 0.9, 1.0),
            tt.utils.MoveDim(-1, -3),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 16),  # hardcoded for 10x10 input (10-3+1=8)
            snn.RLIB(16, 0.9, 0.9, 1.0),
            nn.Linear(16, n_labels),
            snn.LI(n_labels, 0.9),
        )

    def forward(self, x):
        return self.mlp(x)


def _outputs_from_fresh_state(model, input_data, num_timesteps=10):
    model.eval()
    model.reset_states()
    outputs = []
    with torch.no_grad():
        for _ in range(num_timesteps):
            output = model(input_data)
            outputs.append(output.detach().clone())
    return outputs


def _assert_trajectories_equal(outputs1, outputs2, tolerance=1e-5):
    assert len(outputs1) == len(outputs2)
    for out1, out2 in zip(outputs1, outputs2):
        torch.testing.assert_close(out1, out2, atol=tolerance, rtol=1e-5)


def test_cnn_snn_compilation_preserves_timestep_outputs(device):
    b, c, h, w = 2, 3, 10, 10
    model = CNNModel(c, b).to(device)
    random_feature = torch.rand(b, c, h, w).to(device)

    # 1. Baseline
    baseline_outputs = _outputs_from_fresh_state(model, random_feature)

    # 2. Compile
    model.compile_parameters()
    compiled_outputs = _outputs_from_fresh_state(model, random_feature)
    _assert_trajectories_equal(baseline_outputs, compiled_outputs)

    # 3. Decompile
    model.decompile_parameters()
    decompiled_outputs = _outputs_from_fresh_state(model, random_feature)
    _assert_trajectories_equal(baseline_outputs, decompiled_outputs)


def test_cnn_snn_state_checkpoint_resumes_next_timestep(device):
    b, c, h, w = 2, 3, 10, 10
    model = CNNModel(c, b).to(device)
    random_feature = torch.rand(b, c, h, w).to(device)

    model.eval()
    model.reset_states()
    for _ in range(5):
        _ = model(random_feature)

    states = model.save_states()
    assert len(states) > 0

    model2 = CNNModel(c, b).to(device)
    model2.load_state_dict(model.state_dict())
    model2.load_states(states, strict=False, device=device)

    # The restored state must continue the same trajectory.
    out1 = model(random_feature)
    out2 = model2(random_feature)
    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("layer_class", SNN_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("shape", [(4,), (2, 4), (2, 3, 4)])
def test_all_snn_compiled_trajectories_match_dynamic_parameters(layer_class, shape):
    layer = layer_class(4)
    inputs = [torch.randn(shape) for _ in range(3)]
    baseline = [layer(x) for x in inputs]
    assert all(output.shape == shape for output in baseline)
    assert all(getattr(layer, name).shape == shape for name in layer._state_names)
    raw_parameters = list(layer.parameters())
    layer.reset_states()
    layer.compile_parameters()
    compiled = [layer(x.detach()) for x in inputs]
    for expected, actual in zip(baseline, compiled):
        torch.testing.assert_close(actual, expected)
    layer.decompile_parameters()
    assert all(a is b for a, b in zip(raw_parameters, layer.parameters()))
    layer.reset_states()
    decompiled = [layer(x.detach()) for x in inputs]
    for expected, actual in zip(baseline, decompiled):
        torch.testing.assert_close(actual, expected)
