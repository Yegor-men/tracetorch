from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from .. import functional


class SetupMixin:
    """A mixin helper class, used exclusively to initialize parameters."""

    def _register_parameter(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
            inverse_function=lambda x: x
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
        # param_rank = param_tensor.ndim
        # setattr(self, f"{name}_rank", param_rank)
        if learnable:
            setattr(self, f"raw_{name}", nn.Parameter(param_tensor.detach().clone()))
        else:
            self.register_buffer(f"raw_{name}", param_tensor.detach().clone())

    def _register_decay(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        self._register_parameter(name, value, rank, learnable, inverse_function=functional.sigmoid_inverse)

    def _register_threshold(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
    ):
        self._register_parameter(name, value, rank, learnable, inverse_function=functional.softplus_inverse)
