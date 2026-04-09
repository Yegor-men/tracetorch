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

        # Store compilation metadata for decompile
        self._compile_metadata = {}

        # Find all raw_* parameters and their corresponding properties
        for attr_name in list(self.__dict__.keys()):
            if attr_name.startswith('raw_'):
                param_name = attr_name[4:]  # Remove 'raw_' prefix

                # Get current computed value
                computed_value = getattr(self, param_name)

                # Store metadata for decompile
                self._compile_metadata[param_name] = {
                    'raw_value': getattr(self, attr_name).detach().clone(),
                    'is_parameter': isinstance(getattr(self, param_name), nn.Parameter),
                    'learnable': hasattr(self, f'learn_{param_name}') and getattr(self, f'learn_{param_name}')
                }

                # Replace raw parameter with pre-computed buffer
                delattr(self, attr_name)
                self.register_buffer(param_name, computed_value.detach().clone())

                # Remove the property (will be replaced by direct attribute access)
                if hasattr(self.__class__, param_name):
                    delattr(self.__class__, param_name)

        self._compiled = True

    def TTdecompile(self):
        """Decompile layer to restore training capabilities"""
        if not hasattr(self, '_compiled') or not self._compiled:
            return

        # Restore raw_* parameters from metadata
        for param_name, metadata in self._compile_metadata.items():
            # Remove the compiled buffer
            if hasattr(self, param_name):
                delattr(self, param_name)

            # Restore raw parameter
            raw_name = f"raw_{param_name}"
            if metadata['learnable']:
                setattr(self, raw_name, nn.Parameter(metadata['raw_value']))
            else:
                self.register_buffer(raw_name, metadata['raw_value'])

            # Recreate the property (this is tricky - need to recreate the original property)
            # We'll need to store the activation function type during initial registration

        # Clean up
        delattr(self, '_compiled')
        delattr(self, '_compile_metadata')
