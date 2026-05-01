from typing import TypedDict, Optional, Literal, Union, Dict, Any, Set
import torch
from torch import nn
from .. import functional


class Layer(nn.Module):
    r"""The superclass used for all traceTorch layers.
    Handles state management, parameter initialization, compilation and decompilation, moving around tensors to the target dimension.

    Args:
        num_neurons (int): the number of neurons the layer is considered to have. When initializing any hidden states or registering parameters via the tracetorch methods, this is the value used.
        dim (int, default=-1): the dimension along which the layer operates.
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
            inverse_fn=lambda x: x,
            activation_fn=lambda x: x,
    ) -> None:
        r"""Register a parameter with dynamic activation functions.

        Creates a raw parameter that can be dynamically transformed through activation functions.
        The raw parameter is stored as ``raw_{name}`` while the activated version is accessed via ``name``.

        Args:
            name (str): parameter name. Access raw version via ``self.raw_{name}``, activated via ``self.{name}``.
            value (Union[float, torch.Tensor]): initial value. Scalar or vector matching `num_neurons` if rank=1. Can be set to a custom PyTorch tensor instead, and will automatically update the rank depending on the tensor's rank.
            rank (Literal[0, 1]): 0 for scalar, 1 for vector of length `num_neurons`.
            learnable (bool): whether parameter should be trainable (nn.Parameter) or fixed (buffer).
            init_fn (Callable): function applied once during initialization to create raw parameter.
            inverse_fn (Callable): function used during decompilation to recover raw parameter.
            activation_fn (Callable): function applied when accessing the parameter dynamically.

        Notes:
            - Raw parameters are stored as ``nn.Parameter`` if learnable, otherwise as buffers.
            - Dynamic access via ``self.{name}`` applies ``activation_fn`` to raw value.
            - Parameters registered this way can be compiled and decompiled by traceTorch by utilizing the ``activation_fn`` and ``inverse_fn``.
        """

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
        self._dynamic_params[name] = activation_fn
        self._inverse_functions[name] = inverse_fn

    def __getattr__(self, name: str):
        r"""Intercept attribute access to dynamically compute activations on raw parameters."""
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

    def _initialize_state(self, state_name: str) -> None:
        r"""Initialize and register a state for traceTorch operations.

        Args:
            state_name (str): state name.
        """
        self._state_names.add(state_name)
        setattr(self, state_name, None)

    def _detach_state(self, state_name: str) -> None:
        r"""Detach a state tensor from the computation graph if it exists and is not None.

        Args:
            state_name (str): state name.
        """
        state = getattr(self, state_name)
        if state is not None:
            setattr(self, state_name, state.detach())

    def detach_states(self) -> None:
        r"""Detach all initialized state tensors from the computation graph if they are not None."""
        for state_name in self._state_names:
            self._detach_state(state_name)

    def _zero_state(self, state_name: str) -> None:
        r"""Set a state to None.

        Args:
            state_name (str): state name.
        """
        setattr(self, state_name, None)

    def zero_states(self) -> None:
        r"""Set all initialized states to None."""
        for state_name in self._state_names:
            self._zero_state(state_name)

    def _ensure_state(self, state_name: str, reference_tensor: torch.Tensor) -> None:
        r"""Initialize a state with zeros if it is None.

        Args:
            state_name (str): state name.
            reference_tensor (torch.Tensor): the reference tensor, whose shape the state will copy. The shape will be the same except ``dim``, which will be set to ``num_neurons`` instead.
        """
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

    def _ensure_states(self, reference_tensor: torch.Tensor) -> None:
        r"""Initialize all initialized states with zeros if they are None.

        Args:
            reference_tensor (torch.Tensor): the reference tensor, whose shape the states will copy. The shapes will be the same except ``dim``, which will be set to ``num_neurons`` instead.
        """
        for state_name in self._state_names:
            self._ensure_state(state_name, reference_tensor)

    def _to_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Move a tensor's ``dim`` dimension to the working (last) dimension.

        Args:
            tensor (torch.Tensor): the tensor whose ``dim`` dimension will be moved to the last dimension.
        """
        return tensor.movedim(self.dim, -1)

    def _from_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Move tensor back from the working (last) dimension to the ``dim`` dimension.

        Args:
            tensor (torch.Tensor): the tensor whose last dimension will be moved to the ``dim`` dimension.
        """
        return tensor.movedim(-1, self.dim)

    def TTcompile(self) -> None:
        r"""Compile the layer for inference by pre-computing parameters.

        Notes:
            All parameters registered via ``_register_parameter`` will be optimized, as the ``activation_fn`` will be baked in to the activated parameter.
            Proper training is not possible on a compiled layer.
        """
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

    def TTdecompile(self) -> None:
        r"""Decompile the layer to restore training capabilities.

        Notes:
            All parameters registered via ``_register_parameter`` will be decompiled, as the ``inverse_fn`` will be used to re-create the raw version of the parameter.
        """
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
