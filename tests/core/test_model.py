"""Recursive model operations and hidden-state checkpoint contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tracetorch as tt


class StatefulLayer(tt.Layer):
    """Minimal stateful module to test core operations without SNN/RNN behavior."""

    def __init__(self, names=("mem",)):
        super().__init__()
        self.define_parameter("gain", torch.ones(4))
        self.names = names
        for name in names:
            self.define_state(name, (4,))

    def forward(self, x):
        self.zero_states(x)
        for name in self.names:
            setattr(self, name, 0.5 * getattr(self, name) + self.gain * x)
        return getattr(self, self.names[0])


def test_state_checkpoints_are_unique_independent_and_separate_from_weights():
    model = tt.Model()
    model.net = nn.Sequential(StatefulLayer(("mem", "prev_output")), StatefulLayer(("H", "C")))
    model.alias = model.net[0]
    model.cycle = {"model": model}
    model.net(torch.ones(2, 4))
    model.compile_parameters()
    states = model.save_states()
    assert set(states) == {"net.0.mem", "net.0.prev_output", "net.1.H", "net.1.C"}
    assert set(model.state_dict()) == {"net.0.raw_gain", "net.1.raw_gain", "alias.raw_gain"}
    for name in states:
        assert name not in model.state_dict()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    model.reset_states()
    model.load_states(states)
    torch.testing.assert_close(model.net[0].mem, states["net.0.mem"])
    model.net[0].mem.add_(1)
    assert not torch.equal(model.net[0].mem, states["net.0.mem"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key])
    with pytest.raises(ValueError, match="extra"):
        model.load_states({**states, "net.0.raw_gain": torch.ones(4)})
    with pytest.raises(ValueError, match="missing"):
        model.load_states({})
    model.reset_states()
    with pytest.raises(ValueError, match="Shape mismatch"):
        model.load_states({**states, "net.0.mem": torch.zeros(2, 3)})
    model.load_states(states)
    model.load_state_dict(weights)
    torch.testing.assert_close(model.net[0].mem, states["net.0.mem"])


def test_model_visits_plain_containers_shared_layers_and_cycles_once():
    class CountingLayer(StatefulLayer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def compile_parameters(self):
            self.calls += 1
            super().compile_parameters()

    layer = CountingLayer()
    model = tt.Model()
    model.bag = SimpleNamespace(layers=[layer, {"same": layer}])
    model.bag.model = model
    layer(torch.ones(2, 4))
    model.compile_parameters()
    assert layer.calls == 1
    states = model.save_states()
    assert set(states) == {"bag.layers[0].mem"}
    model.detach_states()
    assert not layer.mem.requires_grad
    model.reset_states()
    model.load_states(states)
    assert layer.mem.shape == (2, 4)


def test_model_overrides_can_delegate_to_super_without_redispatch():
    class CountingLayer(StatefulLayer):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def compile_parameters(self):
            self.calls += 1
            super().compile_parameters()

    class CustomModel(tt.Model):
        def __init__(self):
            super().__init__()
            self.layer = CountingLayer()
            self.calls = 0

        def compile_parameters(self):
            self.calls += 1
            super().compile_parameters()

    root = CustomModel()
    root.child = CustomModel()
    root.compile_parameters()
    assert root.calls == root.child.calls == 1
    assert root.layer.calls == root.child.layer.calls == 1


def test_save_omits_uninitialized_states_and_owns_detached_snapshots():
    model = tt.Model()
    model.layer = StatefulLayer()
    assert model.save_states() == {}
    model.layer(torch.ones(2, 4))
    saved = model.save_states()
    assert not saved["layer.mem"].requires_grad
    assert saved["layer.mem"].data_ptr() != model.layer.mem.data_ptr()
    with torch.no_grad():
        model.layer.mem.add_(5)
    torch.testing.assert_close(saved["layer.mem"], torch.ones(2, 4))


def test_partial_state_loading_preserves_unspecified_states_and_parameters():
    model = tt.Model()
    model.layer = StatefulLayer(("mem", "other"))
    model.layer(torch.ones(2, 4))
    weight = model.layer.raw_gain.detach().clone()
    model.load_states({"layer.mem": torch.full((2, 4), 3.0),
                       "unrelated": torch.zeros(1)}, strict=False)
    torch.testing.assert_close(model.layer.mem, torch.full((2, 4), 3.0))
    torch.testing.assert_close(model.layer.other, torch.ones(2, 4))
    torch.testing.assert_close(model.layer.raw_gain, weight)


@pytest.mark.parametrize("invalid,error", [(torch.zeros(2, 3), ValueError),
                                         ("not a tensor", TypeError)])
def test_invalid_checkpoint_is_validated_before_any_state_is_changed(invalid, error):
    model = tt.Model()
    model.layer = StatefulLayer(("mem", "other"))
    model.layer(torch.ones(2, 4))
    checkpoint = {"layer.mem": torch.full((2, 4), 3.0), "layer.other": invalid}
    with pytest.raises(error):
        model.load_states(checkpoint)
    torch.testing.assert_close(model.layer.mem, torch.ones(2, 4))
    torch.testing.assert_close(model.layer.other, torch.ones(2, 4))
