from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from tracetorch.snn._ttmodule import TTModule
from tracetorch import functional as tt_functional


class DecayConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    ema: bool
    learnable: bool


class ThresholdConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    surrogate: Any  # has to be a surrogate function
    learnable: bool


class VectorConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    learnable: bool


DEFAULT_ALPHA = {"value": 0.5, "rank": 1, "ema": True, "learnable": True}
DEFAULT_BETA = {"value": 0.5, "rank": 1, "ema": False, "learnable": True}
DEFAULT_GAMMA = {"value": 0.5, "rank": 1, "ema": True, "learnable": True}
DEFAULT_THRESH = {"value": 1.0, "rank": 1, "surrogate": tt_functional.atan_surrogate(2.0), "learnable": True}
DEFAULT_SCALE = {"value": 1.0, "rank": 1, "learnable": True}
DEFAULT_REC_WEIGHT = {"value": 0.0, "rank": 1, "learnable": True}
DEFAULT_BIAS = {"value": 0.0, "rank": 1, "learnable": True}


class SetupHelpers:
    """A mixin helper class, used exclusively to initialize parameters."""

    def _setup_parameter(
            self,
            name: str,
            value: Union[float, torch.Tensor],
            rank: Literal[0, 1],
            learnable: bool,
            inverse_function=torch.as_tensor
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
