from typing import Callable, Optional, Tuple

import torch
from torch import nn


class Layer(nn.Module):
    """Base class for stateful layers and optionally transformed parameters.

    States are ordinary tensor attributes, separate from PyTorch weight
    checkpoints. Each state declares its own trailing shape and input slice.
    Parameters retain their raw nn.Parameter throughout compilation.
    """

    def __init__(self):
        super().__init__()
        self._state_names = set()
        self._state_shapes = {}
        self._state_dims = {}
        self._dynamic_params = {}
        self._parameter_learnability = {}

    def define_parameter(
            self,
            name: str,
            value: torch.Tensor,
            learnable: bool = True,
            initialization_fn: Optional[Callable] = None,
            activation_fn: Optional[Callable] = None,
    ) -> None:
        """Define a raw parameter and its optional runtime transformation.

        value is a tensor of any shape. initialization_fn converts it into the
        raw representation once (e.g. logit for a sigmoid decay).
        self.raw_<name> is always an nn.Parameter; self.<name> returns its
        activated value, or a detached cache while compiled.
        Only the raw parameter is included in state_dict().
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be a torch.Tensor")
        if (not name or "." in name or hasattr(self, name)
                or hasattr(self, f"raw_{name}") or hasattr(self, f"_compiled_{name}")):
            raise ValueError(f"Parameter name is invalid or already in use: {name!r}")

        with torch.no_grad():
            raw_value = value if initialization_fn is None else initialization_fn(value)
            parameter = nn.Parameter(raw_value.detach().clone(), requires_grad=learnable)
        self.register_parameter(f"raw_{name}", parameter)
        self._dynamic_params[name] = activation_fn
        self._parameter_learnability[name] = learnable

    def __getattr__(self, name: str):
        dynamic_params = self.__dict__.get("_dynamic_params", {})
        if name in dynamic_params:
            cache_name = f"_compiled_{name}"
            if cache_name in self._buffers:
                return self._buffers[cache_name]
            raw_value = super().__getattr__(f"raw_{name}")
            activation_fn = dynamic_params[name]
            return raw_value if activation_fn is None else activation_fn(raw_value)
        return super().__getattr__(name)

    def set_parameter_learnable(self, name: str, learnable: bool) -> None:
        """Set intended learnability, including while temporarily compiled.

        Use this instead of changing raw_<name>.requires_grad directly:
        compilation freezes the raw parameter without changing this intention.
        """
        if name not in self._dynamic_params:
            raise KeyError(name)
        self._parameter_learnability[name] = learnable
        raw = getattr(self, f"raw_{name}")
        raw.requires_grad_(learnable and f"_compiled_{name}" not in self._buffers)
        if not raw.requires_grad:
            raw.grad = None

    def define_state(
            self,
            name: str,
            shape: Tuple[int, ...],
            dim: Optional[int] = -1,
    ) -> None:
        """Declare a state, initially None, with its own allocation rule.

        Allocation uses reference.shape[:dim] + shape. Negative dim drops
        trailing reference dimensions; None drops none and appends shape to
        the full reference shape. Zero and positive indices are rejected.
        An empty shape is allowed.
        """
        if not name or "." in name or hasattr(self, name):
            raise ValueError(f"State name is invalid or already in use: {name!r}")
        if not isinstance(shape, tuple) or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in shape):
            raise ValueError("shape must be a tuple of nonnegative integers")
        if dim is not None and (not isinstance(dim, int) or isinstance(dim, bool) or dim >= 0):
            raise ValueError("dim must be a negative integer or None")
        self._state_names.add(name)
        self._state_shapes[name] = shape
        self._state_dims[name] = dim
        setattr(self, name, None)

    def detach_state(self, state_name: str) -> None:
        """Detach an existing state from its computation graph."""
        if state_name not in self._state_names:
            raise KeyError(state_name)
        state = getattr(self, state_name)
        if state is not None:
            setattr(self, state_name, state.detach())

    def detach_states(self) -> None:
        """Detach all states."""
        for state_name in self._state_names:
            self.detach_state(state_name)

    def reset_state(self, state_name: str) -> None:
        """Set a declared state to None for lazy reinitialization."""
        if state_name not in self._state_names:
            raise KeyError(state_name)
        setattr(self, state_name, None)

    def reset_states(self) -> None:
        """Reset all states to None."""
        for state_name in self._state_names:
            self.reset_state(state_name)

    def zero_state(self, state_name: str, reference_tensor: torch.Tensor) -> None:
        """Allocate zeros from a state's shape rule, only if it is None.

        The reference supplies the leading instance dimensions, dtype and
        device. Existing states are not overwritten or silently resized.
        Custom layers may override this method for nonzero initialization.
        """
        if state_name not in self._state_names:
            raise KeyError(state_name)
        if getattr(self, state_name) is None:
            dim = self._state_dims[state_name]
            if dim is not None and -dim > reference_tensor.ndim:
                raise ValueError(f"State {state_name!r} cannot drop {-dim} dimensions "
                                 f"from a {reference_tensor.ndim}-dimensional reference")
            shape = tuple(reference_tensor.shape[:dim]) + self._state_shapes[state_name]
            setattr(self, state_name, reference_tensor.new_zeros(shape))

    def zero_states(self, reference_tensor: torch.Tensor) -> None:
        """Lazily allocate all states using the same reference tensor."""
        for state_name in self._state_names:
            self.zero_state(state_name, reference_tensor)

    def compile_parameter(self, name: str) -> None:
        """Cache an activated parameter for inference and freeze its raw value.

        The cache is a nonpersistent buffer. Raw parameter identity and intended
        learnability are preserved; repeated compilation is a no-op.
        """
        if name not in self._dynamic_params:
            raise KeyError(name)
        cache_name = f"_compiled_{name}"
        if cache_name in self._buffers:
            return
        with torch.no_grad():
            cache = getattr(self, name).detach().clone()
        self.register_buffer(cache_name, cache, persistent=False)
        raw = getattr(self, f"raw_{name}")
        raw.requires_grad_(False)
        raw.grad = None

    def compile_parameters(self) -> None:
        """Compile every parameter defined through define_parameter."""
        for name in self._dynamic_params:
            self.compile_parameter(name)

    def decompile_parameter(self, name: str) -> None:
        """Delete a parameter cache and restore intended learnability."""
        if name not in self._dynamic_params:
            raise KeyError(name)
        cache_name = f"_compiled_{name}"
        if cache_name not in self._buffers:
            return
        delattr(self, cache_name)
        getattr(self, f"raw_{name}").requires_grad_(self._parameter_learnability[name])

    def decompile_parameters(self) -> None:
        """Decompile every parameter defined through define_parameter."""
        for name in self._dynamic_params:
            self.decompile_parameter(name)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Covers both layer.load_state_dict() and loading through a parent model.
        # A cache computed from the old weights must never survive a weight load.
        self.decompile_parameters()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
