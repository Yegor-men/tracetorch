from ._leaky_integrator import LeakyIntegrator
from ._leaky_integrator import DEFAULT_ALPHA
from ._leaky_integrator import DEFAULT_BETA
# from ._leaky_integrator import DEFAULT_GAMMA
from ._leaky_integrator import DEFAULT_POS_THRESH
# from ._leaky_integrator import DEFAULT_NEG_THRESH
# from ._leaky_integrator import DEFAULT_WEIGHT
# from ._leaky_integrator import DEFAULT_BIAS

from typing import Union, Literal, Any
import torch


class SLIF(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = DEFAULT_ALPHA["value"],
            beta: Union[float, torch.Tensor] = DEFAULT_BETA["value"],
            pos_threshold: Union[float, torch.Tensor] = DEFAULT_POS_THRESH["value"],
            dim: int = -1,
            alpha_rank: Literal[0, 1] = DEFAULT_ALPHA["rank"],
            beta_rank: Literal[0, 1] = DEFAULT_BETA["rank"],
            pos_threshold_rank: Literal[0, 1] = DEFAULT_POS_THRESH["rank"],
            alpha_ema: bool = DEFAULT_ALPHA["use_averaging"],
            beta_ema: bool = DEFAULT_BETA["use_averaging"],
            learn_alpha: bool = DEFAULT_ALPHA["learnable"],
            learn_beta: bool = DEFAULT_BETA["learnable"],
            learn_pos_threshold: bool = DEFAULT_POS_THRESH["learnable"],
            surrogate_derivative: Any = DEFAULT_POS_THRESH["surrogate"],
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
            "use_averaging": beta_ema,
            "learnable": learn_beta,
        }

        pos_threshold_setup = {
            "value": pos_threshold,
            "rank": pos_threshold_rank,
            "surrogate": surrogate_derivative,
            "learnable": learn_pos_threshold,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            alpha_setup=alpha_setup,
            beta_setup=beta_setup,
            gamma_setup=None,
            pos_threshold_setup=pos_threshold_setup,
            neg_threshold_setup=None,
            weight_setup=None,
            bias_setup=None,
        )
