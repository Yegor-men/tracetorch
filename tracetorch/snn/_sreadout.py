from ._leaky_integrator import LeakyIntegrator
from ._leaky_integrator import DEFAULT_ALPHA
from ._leaky_integrator import DEFAULT_BETA
# from ._leaky_integrator import DEFAULT_GAMMA
# from ._leaky_integrator import DEFAULT_POS_THRESH
# from ._leaky_integrator import DEFAULT_NEG_THRESH
# from ._leaky_integrator import DEFAULT_POS_SCALE
# from ._leaky_integrator import DEFAULT_NEG_SCALE
# from ._leaky_integrator import DEFAULT_WEIGHT
# from ._leaky_integrator import DEFAULT_BIAS

from typing import Union, Literal, Any
import torch


class SReadout(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = DEFAULT_ALPHA["value"],
            beta: Union[float, torch.Tensor] = DEFAULT_BETA["value"],
            dim: int = -1,
            alpha_rank: Literal[0, 1] = DEFAULT_ALPHA["rank"],
            beta_rank: Literal[0, 1] = DEFAULT_BETA["rank"],
            alpha_ema: bool = DEFAULT_ALPHA["use_averaging"],
            learn_alpha: bool = DEFAULT_ALPHA["learnable"],
            learn_beta: bool = DEFAULT_BETA["learnable"],
    ):
        alpha_setup = {
            "value": alpha,
            "rank": alpha_rank,
            "use_averaging": alpha_ema,
            "learnable": learn_alpha,
        }

        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "use_averaging": True,
            "learnable": learn_beta,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            alpha_setup=alpha_setup,
            beta_setup=beta_setup,
        )
