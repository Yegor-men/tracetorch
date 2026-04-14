from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional


class Layer(nn.Module):
    """
    Universal base mixin helper for all recurrent layers.

    Handles state management, movedim, parameter registration, compile/decompile
    """

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
            init_fn=lambda x: x,
            inverse_function=lambda x: x,
            activation_function=lambda x: x,
    ):
        if not hasattr(self, '_dynamic_params'):
            self._dynamic_params = {}
            self._inverse_functions = {}

        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                pass
            elif value.ndim == 1:
                assert value.numel() == self.num_neurons, f"{name} does not have {self.num_neurons} elements"
            else:
                raise ValueError(f"rank (.ndim) of provided {name} is not 0 (scalar) or 1 (vector)")
            param_tensor = init_fn(value)
        else:
            value = float(value)
            if rank == 0:
                param_tensor = init_fn(torch.tensor(value))
            elif rank == 1:
                param_tensor = init_fn(torch.full([self.num_neurons], value))
            else:
                raise ValueError(f"{name} rank is not 0 (scalar) or 1 (vector)")

        # save the raw_ variant of the parameter as an nn.Parameter or buffer
        if learnable:
            setattr(self, f"raw_{name}", nn.Parameter(param_tensor.detach().clone()))
        else:
            self.register_buffer(f"raw_{name}", param_tensor.detach().clone())

        # Store the activation and inverse functions directly in the instance
        self._dynamic_params[name] = activation_function
        self._inverse_functions[name] = inverse_function

    def __getattr__(self, name: str):
        """Intercept attribute access to dynamically compute activations on raw parameters."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if '_dynamic_params' in self.__dict__ and name in self._dynamic_params:
                raw_name = f"raw_{name}"
                try:
                    raw_val = super().__getattr__(raw_name)
                    return self._dynamic_params[name](raw_val)
                except AttributeError:
                    pass
            raise

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
        """Initialize a state with zeros if None, otherwise return existing state"""
        state = getattr(self, state_name)
        if state is None:
            # Create shape that matches reference_tensor except for self.dim
            shape = list(reference_tensor.shape)
            shape[self.dim] = self.num_neurons  # Set the target dimension to num_neurons

            state = torch.zeros(
                shape,
                dtype=reference_tensor.dtype,
                device=reference_tensor.device,
            )
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

    def TTcompile(self):
        """Compile layer for inference by pre-computing parameters"""
        if hasattr(self, '_compiled') and self._compiled:
            return

        self._compile_metadata = {}
        if not hasattr(self, '_dynamic_params'):
            return

        for param_name in list(self._dynamic_params.keys()):
            raw_name = f"raw_{param_name}"
            try:
                raw_tensor = super().__getattr__(raw_name)
            except AttributeError:
                continue

            # Get current computed value via our __getattr__ interceptor
            computed_value = getattr(self, param_name)
            is_parameter = isinstance(raw_tensor, nn.Parameter)

            self._compile_metadata[param_name] = {
                'is_parameter': is_parameter,
                'learnable': raw_tensor.requires_grad if is_parameter else False
            }

            # Delete the raw attribute natively
            delattr(self, raw_name)

            # By registering a buffer, PyTorch handles its persistence natively.
            # Our __getattr__ will organically ignore it because super().__getattr__(param_name) will now succeed.
            self.register_buffer(param_name, computed_value.detach().clone())

        self._compiled = True

    def TTdecompile(self):
        """Decompile layer to restore training capabilities"""
        if not hasattr(self, '_compiled') or not self._compiled:
            return

        for param_name, metadata in self._compile_metadata.items():
            if not hasattr(self, param_name):
                continue

            # The currently stored computed buffer
            compiled_value = getattr(self, param_name)

            # Re-convert to raw via the inverse function
            inverse_fn = self._inverse_functions[param_name]
            raw_value = inverse_fn(compiled_value)

            # Eliminate compiled buffer
            delattr(self, param_name)

            raw_name = f"raw_{param_name}"
            if metadata['is_parameter']:
                self.register_parameter(raw_name,
                                        nn.Parameter(raw_value.detach().clone(), requires_grad=metadata['learnable']))
            else:
                self.register_buffer(raw_name, raw_value.detach().clone())

        delattr(self, '_compiled')
        delattr(self, '_compile_metadata')
