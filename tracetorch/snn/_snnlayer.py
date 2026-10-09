from typing import Literal, Union

import torch
from torch import nn

from .. import inverse_fn
from ..core import Layer as BaseLayer


class Layer(BaseLayer):
    """SNN parameter helpers for scalar or per-neuron initialization.

    Concrete SNN layers set num_features themselves. Numeric values with rank 1
    are expanded per neuron; rank 0 shares a scalar. Tensor values are used
    directly after checking they are scalar or per-neuron vectors.
    All built-in SNN layers operate on the final input dimension.
    """

    def _parameter_tensor(
            self, name: str, value: Union[float, torch.Tensor], rank: Literal[0, 1],
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            if value.ndim == 1 and value.numel() == self.num_features:
                return value
            raise ValueError(f"{name} must be a scalar or vector of length {self.num_features}")
        if rank == 0:
            return torch.tensor(float(value))
        if rank == 1:
            return torch.full((self.num_features,), float(value))
        raise ValueError(f"{name} rank must be 0 (scalar) or 1 (per-neuron)")

    def define_decay(self, name: str, value: Union[float, torch.Tensor],
                     rank: Literal[0, 1], learnable: bool) -> None:
        """Define a scalar or per-neuron decay constrained to (0, 1)."""
        self.define_parameter(
            name, self._parameter_tensor(name, value, rank), learnable,
            initialization_fn=inverse_fn.sigmoid, activation_fn=torch.sigmoid,
        )

    def define_threshold(self, name: str, value: Union[float, torch.Tensor],
                         rank: Literal[0, 1], learnable: bool) -> None:
        """Define a scalar or per-neuron positive threshold."""
        self.define_parameter(
            name, self._parameter_tensor(name, value, rank), learnable,
            initialization_fn=inverse_fn.softplus, activation_fn=nn.functional.softplus,
        )

    def define_unbound_parameter(self, name: str, value: Union[float, torch.Tensor],
                                 rank: Literal[0, 1], learnable: bool) -> None:
        """Define an unconstrained SNN parameter, such as a scale or weight."""
        self.define_parameter(name, self._parameter_tensor(name, value, rank), learnable)
