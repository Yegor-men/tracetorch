from typing import TypedDict, Optional, Literal, Union, Dict, Any
from ._leaky_integrator import LeakyIntegrator
import torch
from torch import nn

from ._leaky_integrator import DEFAULT_ALPHA
from ._leaky_integrator import DEFAULT_BETA
from ._leaky_integrator import DEFAULT_GAMMA
from ._leaky_integrator import DEFAULT_THRESHOLD
from ._leaky_integrator import DEFAULT_SCALE
from ._leaky_integrator import DEFAULT_REC_WEIGHT
from ._leaky_integrator import DEFAULT_BIAS


class LI(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": True,
            "learnable": learn_beta,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_beta_setup=beta_setup,
        )
