from typing import Dict

import torch
from torch import nn

from ._layer import Layer


class Model(nn.Module):
    """Manage hidden states and parameter caches across a model hierarchy.

    Weight checkpoints use native PyTorch state_dict(). Hidden-state
    checkpoints use save_states()/load_states() and contain no parameters.
    """

    def _walk_objects(self, stop_at=None):
        """Visit each object once, preferring registered module paths.

        Plain Python containers and object attributes are also supported.
        Shared layers and circular references do not produce duplicate paths.
        """
        visited = set()

        def walk(obj, path):
            if id(obj) in visited:
                return
            visited.add(id(obj))
            yield path, obj
            if stop_at is not None and stop_at(obj):
                return

            if isinstance(obj, nn.Module):
                for name, child in obj._modules.items():
                    if child is not None:
                        child_path = f"{path}.{name}" if path else name
                        yield from walk(child, child_path)

            if isinstance(obj, dict):
                for key, value in obj.items():
                    yield from walk(value, f"{path}[{key}]")
            elif isinstance(obj, (list, tuple, set)):
                for index, value in enumerate(obj):
                    yield from walk(value, f"{path}[{index}]")
            elif not isinstance(obj, torch.Tensor) and (isinstance(obj, nn.Module) or not callable(obj)):
                for name, value in getattr(obj, "__dict__", {}).items():
                    # PyTorch internals hold parameters, buffers, hooks, and
                    # submodules already visited above, not additional layers.
                    if isinstance(obj, nn.Module) and name.startswith("_"):
                        if name in ("_modules", "_parameters", "_buffers") or "hook" in name:
                            continue
                    if isinstance(value, (torch.Tensor, str, bytes, int, float, bool, type(None))):
                        continue
                    child_path = f"{path}.{name}" if path else name
                    yield from walk(value, child_path)

        yield from walk(self, "")

    def _call_recursive(self, method_name: str) -> None:
        """Call a layer operation once per object, respecting model overrides."""
        def overrides_operation(obj):
            return (isinstance(obj, Model) and obj is not self
                    and getattr(type(obj), method_name, None) is not getattr(Model, method_name, None))

        for _, obj in self._walk_objects(stop_at=overrides_operation):
            # A subclass may call super() from its override; do not redispatch
            # that override on the root. Nested overrides own their traversal.
            if obj is self:
                continue
            if isinstance(obj, Model):
                method = getattr(type(obj), method_name, None)
                if method is not None and method is not getattr(Model, method_name, None):
                    method(obj)
            else:
                method = getattr(obj, method_name, None)
                if callable(method):
                    method()

    def _state_slots(self):
        for path, obj in self._walk_objects():
            if isinstance(obj, Layer):
                for name in sorted(obj._state_names):
                    full_name = f"{path}.{name}" if path else name
                    yield full_name, obj, name

    def save_states(self) -> Dict[str, torch.Tensor]:
        """Return detached copies of initialized states, keyed by layer path.

        None states are omitted. The result can be saved with torch.save();
        neither raw weights nor compiled parameter buffers are included.
        """
        return {
            full_name: getattr(layer, name).detach().clone()
            for full_name, layer, name in self._state_slots()
            if getattr(layer, name) is not None
        }

    def load_states(self, states: Dict[str, torch.Tensor], strict: bool = True, device=None) -> None:
        """Load a hidden-state checkpoint without modifying model parameters.

        Strict mode rejects missing and extra state names. Existing shapes and
        declared trailing shapes are checked. Loaded tensors are detached and
        copied, so subsequent updates do not mutate the checkpoint dictionary.
        Device defaults to the layer's parameters, buffers, or saved tensor.
        """
        slots = list(self._state_slots())
        expected = {full_name for full_name, _, _ in slots}
        if strict:
            missing = expected - states.keys()
            extra = states.keys() - expected
            if missing or extra:
                raise ValueError(f"State keys do not match: missing={sorted(missing)}, extra={sorted(extra)}")

        # Validate everything before changing any state.
        for full_name, layer, name in slots:
            if full_name not in states:
                continue
            tensor = states[full_name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"State {full_name!r} must be a torch.Tensor")
            shape = layer._state_shapes[name]
            if shape and tuple(tensor.shape[-len(shape):]) != shape:
                raise ValueError(f"Shape mismatch for {full_name}: expected trailing {shape}, got {tensor.shape}")
            current = getattr(layer, name)
            if current is not None and current.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {full_name}: expected {current.shape}, got {tensor.shape}")

        for full_name, layer, name in slots:
            if full_name not in states:
                continue
            tensor = states[full_name]
            target_device = device
            if target_device is None:
                parameter = next(layer.parameters(), None)
                buffer = next(layer.buffers(), None)
                target_device = (parameter.device if parameter is not None else
                                 buffer.device if buffer is not None else tensor.device)
            setattr(layer, name, tensor.detach().to(target_device).clone())

    def reset_states(self) -> None:
        """Set all hidden states to None for lazy initialization next forward."""
        self._call_recursive("reset_states")

    def detach_states(self) -> None:
        """Detach every hidden state while preserving its numerical value."""
        self._call_recursive("detach_states")

    def compile_parameters(self) -> None:
        """Cache activated traceTorch parameters across the model for inference."""
        self._call_recursive("compile_parameters")

    def decompile_parameters(self) -> None:
        """Delete caches and restore intended parameter learnability."""
        self._call_recursive("decompile_parameters")
