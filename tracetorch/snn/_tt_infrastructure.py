from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional


class TTLayer(nn.Module):
    """A mixin helper class, used to help manage parameters and hidden states."""

    def __init__(self, num_neurons: int, dim: int = -1):
        super().__init__()
        self._state_names = set()
        self.num_neurons = num_neurons
        self.dim = dim

    def _register_parameter(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
            inverse_function=lambda x: x,
            activation_function=lambda x: x,
    ):
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                pass
            elif value.ndim == 1:
                assert value.numel() == self.num_neurons, f"{name} does not have {self.num_neurons} elements"
            else:
                raise ValueError(f"rank (.ndim) of provided {name} is not 0 (scalar) or 1 (vector)")
            param_tensor = inverse_function(value)
        else:
            value = float(value)
            if rank == 0:
                param_tensor = inverse_function(torch.tensor(value))
            elif rank == 1:
                param_tensor = inverse_function(torch.full([self.num_neurons], value))
            else:
                raise ValueError(f"{name} rank is not 0 (scalar) or 1 (vector)")

        # save the raw_ variant of the parameter as an nn.Parameter or buffer
        if learnable:
            setattr(self, f"raw_{name}", nn.Parameter(param_tensor.detach().clone()))
        else:
            self.register_buffer(f"raw_{name}", param_tensor.detach().clone())

        # create a @property of the raw_ parameter that passes it through the respective activation function
        setattr(self.__class__, name, self._make_property(name, activation_function))

    @staticmethod
    def _make_property(name, activation_function):
        def getter(self):
            return activation_function(getattr(self, f"raw_{name}"))

        return property(getter)

    def _register_decay(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        self._register_parameter(
            name,
            value,
            rank,
            learnable,
            inverse_function=functional.sigmoid_inverse,
            activation_function=nn.functional.sigmoid,
        )

    def _register_threshold(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        self._register_parameter(
            name,
            value,
            rank,
            learnable,
            inverse_function=functional.softplus_inverse,
            activation_function=nn.functional.softplus,
        )

    def _initialize_state(self, state_name: str):
        """Initialize and register a state name for bulk operations"""
        self._state_names.add(state_name)
        setattr(self, state_name, None)

    def _detach_state(self, state_name: str):
        """Detach a state tensor if it exists"""
        state = getattr(self, state_name)
        if state is not None:
            setattr(self, state_name, state.detach())

    def detach_states(self):
        """Detach all registered states"""
        for state_name in self._state_names:
            self._detach_state(state_name)

    def _zero_state(self, state_name: str):
        """Set a state to None"""
        setattr(self, state_name, None)

    def zero_states(self):
        """Zero all registered states"""
        for state_name in self._state_names:
            self._zero_state(state_name)

    def _ensure_state(self, state_name: str, reference_tensor: torch.Tensor):
        """Initialize a state with zeros_like if None, otherwise return existing state"""
        state = getattr(self, state_name)
        if state is None:
            state = torch.zeros_like(reference_tensor)
            setattr(self, state_name, state)

    def _ensure_states(self, reference_tensor: torch.Tensor):
        """Ensure all registered states are initialized"""
        for state_name in self._state_names:
            self._ensure_state(state_name, reference_tensor)

    def _to_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to working dimension (last dim)"""
        return tensor.movedim(self.dim, -1)

    def _from_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor back from working dimension"""
        return tensor.movedim(-1, self.dim)


class TTModel(nn.Module):
    """
    Base class that makes it trivial to call lifecycle methods (zero_states, detach_states, ...)
    across an entire model tree (registered submodules *and* common non-module containers).
    Inherit your models from this to get model.zero_states() / model.detach_states() behavior.
    """

    def __init__(self):
        super().__init__()

    def get_param_count(self) -> int:
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

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

            # 1) Handle TTLayer instances (leaf components)
            if isinstance(obj, TTLayer):
                fn = getattr(obj, method_name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        fn()

            # 2) Handle TTModel instances that override the method
            elif isinstance(obj, TTModel):
                cls_fn = getattr(obj.__class__, method_name, None)
                base_fn = getattr(TTModel, method_name, None)
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
