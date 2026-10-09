"""Contracts for generic parameter and state management."""

import io

import pytest
import torch
from torch import nn

import tracetorch as tt


def _decay_layer(beta, learnable=True):
    layer = tt.Layer()
    layer.define_parameter("beta", torch.full((4,), beta), learnable,
                           initialization_fn=torch.logit, activation_fn=torch.sigmoid)
    return layer


@pytest.mark.parametrize("reference_shape,state_shape,dim,expected", [
    ((8,), (5, 4), -1, (5, 4)),
    ((2, 8), (5, 4), -1, (2, 5, 4)),
    ((2, 3, 8), (5, 4), -1, (2, 3, 5, 4)),
    ((2, 3, 6, 8), (5, 4), -2, (2, 3, 5, 4)),
    ((2, 3, 6, 8), (5, 4), -4, (5, 4)),
    ((2, 8), (4,), None, (2, 8, 4)),
    ((2, 8), (), -1, (2,)),
    ((), (4,), None, (4,)),
])
def test_state_allocation(reference_shape, state_shape, dim, expected, device):
    layer = tt.Layer()
    layer.define_state(name="state", shape=state_shape, dim=dim)
    reference = torch.empty(reference_shape, dtype=torch.float64, device=device)
    layer.zero_state("state", reference)
    assert layer.state.shape == expected
    assert layer.state.dtype == reference.dtype
    assert layer.state.device == reference.device
    assert torch.count_nonzero(layer.state) == 0
    assert not layer.state.requires_grad
    assert not layer.state_dict()
    assert not hasattr(layer, "num_features")
    assert not hasattr(layer, "dim")
    assert not hasattr(layer, "to_working_dim")
    assert not hasattr(layer, "from_working_dim")


@pytest.mark.parametrize("dim", [0, 1, 2, -1.5, True])
def test_state_rejects_invalid_dimension(dim):
    with pytest.raises(ValueError):
        tt.Layer().define_state("state", (4,), dim)


@pytest.mark.parametrize("shape", [4, [4], (-1,), (1.5,), (True,)])
def test_state_rejects_invalid_shape(shape):
    with pytest.raises(ValueError):
        tt.Layer().define_state("state", shape)


def test_state_lifecycle_and_independent_rules():
    layer = tt.Layer()
    layer.define_state("mem", (5,))
    layer.define_state("matrix", (5, 4), dim=-2)
    layer.define_state("append", (4,), dim=None)
    reference = torch.empty(2, 3, 8)
    layer.zero_states(reference)
    assert layer.mem.shape == (2, 3, 5)
    assert layer.matrix.shape == (2, 5, 4)
    assert layer.append.shape == (2, 3, 8, 4)

    original = layer.mem
    layer.mem.fill_(7)
    layer.zero_states(torch.empty(9, 8))
    assert layer.mem is original
    assert torch.all(layer.mem == 7)
    layer.mem = torch.ones_like(layer.mem, requires_grad=True) * 2
    layer.detach_state("mem")
    assert not layer.mem.requires_grad
    layer.reset_state("matrix")
    assert layer.matrix is None and layer.mem is not None
    layer.reset_states()
    assert all(getattr(layer, name) is None for name in layer._state_names)
    layer.zero_states(torch.empty(9, 8))
    assert layer.mem.shape == (9, 5)
    assert layer.matrix.shape == (5, 4)
    assert layer.append.shape == (9, 8, 4)


def test_state_rejects_removing_more_dimensions_than_exist():
    layer = tt.Layer()
    layer.define_state("state", (4,), dim=-2)
    with pytest.raises(ValueError):
        layer.zero_states(torch.empty(8))


def test_custom_zero_state_is_used_by_plural_method():
    class RestingLayer(tt.Layer):
        def __init__(self):
            super().__init__()
            self.define_parameter("bias", torch.full((4,), 2.0))
            self.define_state("mem", (4,))

        def zero_state(self, state_name, reference_tensor):
            if getattr(self, state_name) is None:
                super().zero_state(state_name, reference_tensor)
                setattr(self, state_name, getattr(self, state_name) + self.bias)

    layer = RestingLayer()
    layer.zero_states(torch.empty(3, 4))
    assert torch.all(layer.mem == 2)
    layer.mem.sum().backward()
    torch.testing.assert_close(layer.raw_bias.grad, torch.full((4,), 3.0))


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("learnable", [False, True])
def test_parameter_accepts_arbitrary_tensor_shape(shape, learnable):
    layer = tt.Layer()
    value = torch.ones(shape, dtype=torch.float64)
    calls = []

    def initialize(tensor):
        calls.append(tensor)
        return torch.log(tensor)

    layer.define_parameter("value", value, learnable,
                           initialization_fn=initialize, activation_fn=torch.exp)
    assert isinstance(layer.raw_value, nn.Parameter)
    assert layer.raw_value.requires_grad is learnable
    assert layer.raw_value.shape == shape
    assert layer.raw_value.dtype == value.dtype
    torch.testing.assert_close(layer.value, value)
    raw = layer.raw_value
    layer.compile_parameter("value")
    layer.compile_parameter("value")
    assert not raw.requires_grad and raw.grad is None
    assert not layer.value.requires_grad
    assert layer.value.data_ptr() != raw.data_ptr()
    assert len(list(layer.buffers())) == 1
    assert set(layer.state_dict()) == {"raw_value"}
    layer.decompile_parameter("value")
    layer.decompile_parameter("value")
    assert layer.raw_value is raw
    assert raw.requires_grad is learnable
    assert not list(layer.buffers())
    assert len(calls) == 1
    torch.testing.assert_close(layer.value, value)


def test_base_parameter_requires_tensor_without_rank_convenience():
    with pytest.raises(TypeError):
        tt.Layer().define_parameter("value", 0.9)
    with pytest.raises(TypeError):
        tt.Layer().define_parameter("value", torch.tensor(0.9), rank=1)


def test_learnability_changes_survive_compilation_and_singular_operations():
    layer = tt.Layer()
    layer.define_parameter("first", torch.ones(4))
    layer.define_parameter("second", torch.ones(2), learnable=False)
    layer.compile_parameter("first")
    assert not layer.raw_first.requires_grad
    assert "_compiled_second" not in layer._buffers
    layer.set_parameter_learnable("first", False)
    layer.compile_parameters()
    layer.set_parameter_learnable("second", True)
    assert not layer.raw_second.requires_grad
    layer.compile_parameters()
    layer.decompile_parameter("first")
    assert not layer.raw_first.requires_grad
    assert "_compiled_second" in layer._buffers
    layer.decompile_parameters()
    assert layer.raw_second.requires_grad
    layer.set_parameter_learnable("first", True)
    layer.first.sum().backward()
    assert layer.raw_first.grad is not None
    layer.set_parameter_learnable("first", False)
    assert layer.raw_first.grad is None


def test_compilation_clears_gradients_preserves_optimizer_and_freezes_updates():
    layer = tt.Layer()
    layer.define_parameter("weight", torch.ones(4))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1)
    raw = layer.raw_weight
    layer.weight.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    layer.weight.sum().backward()
    assert raw.grad is not None
    layer.compile_parameters()
    assert raw.grad is None
    before = raw.detach().clone()
    optimizer.step()
    torch.testing.assert_close(raw, before)
    layer.decompile_parameters()
    assert optimizer.param_groups[0]["params"][0] is layer.raw_weight is raw
    layer.weight.sum().backward()
    optimizer.step()
    assert not torch.equal(raw, before)


def test_compiled_buffers_follow_dtype_and_are_never_in_weight_checkpoints(device):
    layer = _decay_layer(0.9, learnable=False)
    layer.define_parameter("threshold", torch.ones(4))
    layer.define_state("mem", (4,))
    layer.to(device)
    reference = torch.ones(2, 4, device=device)
    layer.zero_states(reference)
    before = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    layer.compile_parameters()
    assert set(layer.state_dict()) == set(before)
    for name, tensor in layer.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    stream = io.BytesIO()
    torch.save(layer.state_dict(), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    assert set(saved) == {"raw_beta", "raw_threshold"}
    layer.reset_states()
    layer.double()
    assert layer.beta.dtype == layer.raw_beta.dtype == torch.float64
    assert all(buffer.dtype == torch.float64 for buffer in layer.buffers())
    assert layer.beta.device == layer.raw_beta.device == reference.device
    layer.zero_states(torch.ones(2, 4, dtype=torch.float64, device=device))
    assert layer.mem.dtype == torch.float64


@pytest.mark.parametrize("assign", [False, True])
def test_weight_loading_invalidates_caches_through_parent_module(assign):
    model = nn.Sequential(_decay_layer(0.9, learnable=False))
    source = nn.Sequential(_decay_layer(0.5))
    model[0].compile_parameters()
    model.load_state_dict(source.state_dict(), assign=assign)
    assert not list(model[0].buffers())
    assert not model[0].raw_beta.requires_grad
    torch.testing.assert_close(model[0].beta, torch.full((4,), 0.5))
    model[0].compile_parameters()
    torch.testing.assert_close(model[0].beta, torch.full((4,), 0.5))


@pytest.mark.parametrize("method", ["detach_state", "reset_state", "zero_state"])
def test_state_operations_reject_undeclared_names(method):
    layer = tt.Layer()
    args = ("unknown", torch.empty(4)) if method == "zero_state" else ("unknown",)
    with pytest.raises(KeyError, match="unknown"):
        getattr(layer, method)(*args)


@pytest.mark.parametrize("method", ["compile_parameter", "decompile_parameter",
                                    "set_parameter_learnable"])
def test_parameter_operations_reject_undeclared_names(method):
    layer = tt.Layer()
    args = ("unknown", False) if method == "set_parameter_learnable" else ("unknown",)
    with pytest.raises(KeyError, match="unknown"):
        getattr(layer, method)(*args)


@pytest.mark.parametrize("name", ["", "bad.name", "training", "existing"])
@pytest.mark.parametrize("kind", ["parameter", "state"])
def test_definitions_reject_invalid_or_occupied_names(name, kind):
    layer = tt.Layer()
    layer.define_state("existing", (4,))
    with pytest.raises(ValueError):
        if kind == "parameter":
            layer.define_parameter(name, torch.ones(4))
        else:
            layer.define_state(name, (4,))


def test_parameter_definition_owns_a_detached_copy_of_its_initial_value():
    layer = tt.Layer()
    value = torch.arange(6, dtype=torch.float64).reshape(2, 3).requires_grad_()
    layer.define_parameter("weight", value)
    original = value.detach().clone()
    assert layer.raw_weight.is_leaf and layer.raw_weight.grad_fn is None
    with torch.no_grad():
        value.add_(10)
    torch.testing.assert_close(layer.weight, original)
