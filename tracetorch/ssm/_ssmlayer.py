import torch
from torch import nn
from ..core import Layer as BaseLayer
from typing import Literal
from .. import functional


class Layer(BaseLayer):
    def __init__(self, num_neurons: int, dim: int = -1, d_state: int = 1):
        super().__init__(num_neurons, dim)
        self.d_state = d_state

    def _ensure_state(self, state_name: str, reference_tensor: torch.Tensor):
        state = getattr(self, state_name)
        if state is None:
            shape = list(reference_tensor.shape)
            shape[self.dim] = self.num_neurons
            shape.append(self.d_state)

            state = torch.zeros(shape, dtype=reference_tensor.dtype, device=reference_tensor.device)
            setattr(self, state_name, state)

    def _state_to_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        dim_D = self.dim if self.dim >= 0 else self.dim - 1
        return tensor.movedim(dim_D, -2)

    def _state_from_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        dim_D = self.dim if self.dim >= 0 else self.dim - 1
        return tensor.movedim(-2, dim_D)

    def _register_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        self._register_parameter(name, value, rank, learnable, init_fn=functional.mamba_scale,
                                 inverse_function=torch.log, activation_function=torch.exp)

    def _register_log_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        self._register_parameter(name, value, rank, learnable, init_fn=torch.log, inverse_function=torch.log,
                                 activation_function=torch.exp)

    # NEW: Allows custom SSMs to register scales matching d_state size instead of num_neurons
    def _register_state_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        original_num = self.num_neurons
        self.num_neurons = self.d_state
        self._register_scale(name, value, rank, learnable)
        self.num_neurons = original_num

    def _register_log_state_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        original_num = self.num_neurons
        self.num_neurons = self.d_state
        self._register_log_scale(name, value, rank, learnable)
        self.num_neurons = original_num
