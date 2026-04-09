from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional
from ._layer import Layer


class Model(nn.Module):
    """
    Base class that makes it trivial to call lifecycle methods (zero_states, detach_states, ...)
    across an entire model tree (registered submodules *and* common non-module containers).
    Inherit your models from this to get model.zero_states() / model.detach_states() behavior.
    """

    def __init__(self):
        super().__init__()

    def save_states(self) -> Dict[str, torch.Tensor]:
        """
        Save all hidden states from Layers in the model.

        states = model.save_states()
        torch.save(states, "model_states.pt")

        Returns:
            Dictionary mapping layer_state_name -> tensor, compatible with torch.save()
        """
        states = {}

        def collect_states(obj, path=""):
            if isinstance(obj, Layer):
                for state_name in obj._state_names:
                    state_value = getattr(obj, state_name)
                    if state_value is not None:
                        # Use dot notation for unique identification
                        full_name = f"{path}.{state_name}" if path else state_name
                        states[full_name] = state_value.detach().clone()

            # Recurse into submodules and containers (similar to _call_recursive)
            if isinstance(obj, nn.Module):
                for name, child in obj._modules.items():
                    child_path = f"{path}.{name}" if path else name
                    collect_states(child, child_path)

            # Handle container attributes
            try:
                attrs = getattr(obj, "__dict__", {})
            except Exception:
                attrs = {}

            for attr_name, attr_value in attrs.items():
                if attr_value is None:
                    continue
                if isinstance(attr_value, (list, tuple, set)):
                    for i, el in enumerate(attr_value):
                        collect_states(el, f"{path}.{attr_name}[{i}]")
                elif isinstance(attr_value, dict):
                    for key, el in attr_value.items():
                        collect_states(el, f"{path}.{attr_name}[{key}]")
                elif isinstance(attr_value, (nn.Module, object)) and not isinstance(attr_value, Layer):
                    collect_states(attr_value, f"{path}.{attr_name}")

        collect_states(self)
        return states

    def load_states(self, states: Dict[str, torch.Tensor], strict: bool = True, device=None) -> None:
        """
        Load hidden states into Layers.

        states = torch.load("model_states.pt")
        model.load_states(states)

        Args:
            states: Dictionary from save_states() or torch.load()
            strict: If True, raise error for missing/extra states
            device: Target device for loaded states (auto-detected if None)
        """
        loaded_count = 0
        missing_states = []

        def distribute_states(obj, path=""):
            nonlocal loaded_count

            if isinstance(obj, Layer):
                for state_name in obj._state_names:
                    full_name = f"{path}.{state_name}" if path else state_name

                    if full_name in states:
                        state_tensor = states[full_name]
                        current_state = getattr(obj, state_name)

                        # Validate shape if current state exists
                        if current_state is not None:
                            if current_state.shape != state_tensor.shape:
                                raise ValueError(
                                    f"Shape mismatch for {full_name}: "
                                    f"expected {current_state.shape}, got {state_tensor.shape}"
                                )

                        # Use provided device, or detect from parameters/buffers, or fallback to current device
                        target_device = device
                        if target_device is None:
                            # Try to get device from existing parameters
                            for param in obj.parameters():
                                target_device = param.device
                                break
                            else:
                                # Try to get device from existing buffers
                                for buffer in obj.buffers():
                                    target_device = buffer.device
                                    break
                                else:
                                    # Fallback to the state tensor's device
                                    target_device = state_tensor.device

                        # Set the state, ensuring it's on the correct device
                        setattr(obj, state_name, state_tensor.to(target_device))
                        loaded_count += 1
                    else:
                        missing_states.append(full_name)

            # Recurse (same logic as save_states)
            if isinstance(obj, nn.Module):
                for name, child in obj._modules.items():
                    child_path = f"{path}.{name}" if path else name
                    distribute_states(child, child_path)

            try:
                attrs = getattr(obj, "__dict__", {})
            except Exception:
                attrs = {}

            for attr_name, attr_value in attrs.items():
                if attr_value is None:
                    continue
                if isinstance(attr_value, (list, tuple, set)):
                    for i, el in enumerate(attr_value):
                        distribute_states(el, f"{path}.{attr_name}[{i}]")
                elif isinstance(attr_value, dict):
                    for key, el in attr_value.items():
                        distribute_states(el, f"{path}.{attr_name}[{key}]")
                elif isinstance(attr_value, (nn.Module, object)) and not isinstance(attr_value, Layer):
                    distribute_states(attr_value, f"{path}.{attr_name}")

        distribute_states(self)

        if strict and missing_states:
            raise ValueError(f"Missing states for: {missing_states}")

        print(f"Loaded {loaded_count} states")

    def zero_states(self) -> None:
        """Public API — zero any stateful child that implements zero_states()."""
        self._call_recursive("zero_states")

    def detach_states(self) -> None:
        """Public API — detach (stop-gradient) any stateful child that implements detach_states()."""
        self._call_recursive("detach_states")

    # --- internal recursive walker ---
    def _call_recursive(self, method_name: str) -> None:
        visited: Set[int] = set()

        def recurse(obj: Any) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            # 1) Handle Layer instances (leaf components)
            if isinstance(obj, Layer):
                fn = getattr(obj, method_name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        fn()

            # 2) Handle TTModel instances that override the method
            elif isinstance(obj, Model):
                cls_fn = getattr(obj.__class__, method_name, None)
                base_fn = getattr(Model, method_name, None)
                if cls_fn is not None and cls_fn is not base_fn:
                    try:
                        cls_fn(obj)
                    except TypeError:
                        cls_fn(obj)

            # 3) Handle other objects that have the method
            elif hasattr(obj, method_name):
                fn = getattr(obj, method_name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        fn()

            # 4) Recurse into registered submodules (this covers ModuleList, Sequential, etc.)
            if isinstance(obj, nn.Module):
                for child in obj._modules.values():
                    if child is None:
                        continue
                    recurse(child)

            # 5) Recurse into container attributes that might hold modules or other objects with methods.
            #    This covers plain python lists/tuples/dicts/sets assigned as attributes on modules.
            #    We purposely ignore common atomic types (tensors, numbers, strings).
            try:
                attrs = getattr(obj, "__dict__", {})
            except Exception:
                attrs = {}

            for attr in attrs.values():
                if attr is None:
                    continue
                # common containers
                if isinstance(attr, (list, tuple, set)):
                    for el in attr:
                        recurse(el)
                elif isinstance(attr, dict):
                    for el in attr.values():
                        recurse(el)
                else:
                    # If attribute is a module or object, recurse (visited prevents duplicates)
                    if isinstance(attr, (nn.Module, object)):
                        recurse(attr)

        # start recursion from self
        recurse(self)

    def TTcompile(self):
        """Recursively compile all Layer instances for inference"""
        self._call_recursive("TTcompile")

    def TTuncompile(self):
        """Recursively uncompile all TTLayer instances"""
        self._call_recursive("TTuncompile")
