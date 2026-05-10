import torch
from torch import nn
from ..core import Layer as BaseLayer
from typing import Literal
from .. import functional


class Layer(BaseLayer):
    r"""Base class for traceTorch state-space-model layers.

    SSM layers store states with an additional trailing ``d_state`` dimension.
    This base class adapts ``tt.Layer`` state initialization and dimension
    helpers so concrete SSM layers can still operate on an arbitrary feature
    dimension while carrying a per-feature latent state.

    Args:
        num_neurons (int): number of features in the target dimension.
        dim (int, default=-1): dimension along which the layer operates.
        d_state (int, default=1): latent state size per feature.
    """

    def __init__(self, num_neurons: int, dim: int = -1, d_state: int = 1):
        super().__init__(num_neurons, dim)
        self.d_state = d_state

    def _ensure_state(self, state_name: str, reference_tensor: torch.Tensor):
        r"""Initialize an SSM state with an extra trailing ``d_state`` axis."""
        state = getattr(self, state_name)
        if state is None:
            shape = list(reference_tensor.shape)
            shape[self.dim] = self.num_neurons
            shape.append(self.d_state)

            state = torch.zeros(shape, dtype=reference_tensor.dtype, device=reference_tensor.device)
            setattr(self, state_name, state)

    def _state_to_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Move the feature axis of a state tensor to the second-last axis."""
        dim_D = self.dim if self.dim >= 0 else self.dim - 1
        return tensor.movedim(dim_D, -2)

    def _state_from_working_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Move a state tensor's second-last feature axis back to ``dim``."""
        dim_D = self.dim if self.dim >= 0 else self.dim - 1
        return tensor.movedim(-2, dim_D)

    def _register_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        r"""Register a positive exponential scale parameter."""
        self._register_parameter(name, value, rank, learnable, init_fn=functional.mamba_scale,
                                 inverse_fn=torch.log, activation_fn=torch.exp)

    def _register_log_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        r"""Register a scale parameter stored and activated in log space."""
        self._register_parameter(name, value, rank, learnable, init_fn=torch.log, inverse_fn=torch.log,
                                 activation_fn=torch.exp)

    def _register_state_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        r"""Register a scale parameter whose rank-1 size is ``d_state``."""
        original_num = self.num_neurons
        self.num_neurons = self.d_state
        self._register_scale(name, value, rank, learnable)
        self.num_neurons = original_num

    def _register_log_state_scale(self, name: str, value, rank: Literal[0, 1], learnable: bool):
        r"""Register a log-space scale parameter whose rank-1 size is ``d_state``."""
        original_num = self.num_neurons
        self.num_neurons = self.d_state
        self._register_log_scale(name, value, rank, learnable)
        self.num_neurons = original_num
