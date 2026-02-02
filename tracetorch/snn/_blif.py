from ._leaky_integrator import LeakyIntegrator
# from ._leaky_integrator import DEFAULT_ALPHA
from ._leaky_integrator import DEFAULT_BETA
# from ._leaky_integrator import DEFAULT_GAMMA
from ._leaky_integrator import DEFAULT_POS_THRESH
from ._leaky_integrator import DEFAULT_NEG_THRESH
# from ._leaky_integrator import DEFAULT_WEIGHT
# from ._leaky_integrator import DEFAULT_BIAS

from typing import Union, Literal, Any
import torch


class BLIF(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = DEFAULT_BETA["value"],
            pos_threshold: Union[float, torch.Tensor] = DEFAULT_POS_THRESH["value"],
            neg_threshold: Union[float, torch.Tensor] = DEFAULT_NEG_THRESH["value"],
            dim: int = -1,
            beta_rank: Literal[0, 1] = DEFAULT_BETA["rank"],
            pos_threshold_rank: Literal[0, 1] = DEFAULT_POS_THRESH["rank"],
            neg_threshold_rank: Literal[0, 1] = DEFAULT_NEG_THRESH["rank"],
            beta_ema: bool = DEFAULT_BETA["use_averaging"],
            learn_beta: bool = DEFAULT_BETA["learnable"],
            learn_pos_threshold: bool = DEFAULT_POS_THRESH["learnable"],
            learn_neg_threshold: bool = DEFAULT_NEG_THRESH["learnable"],
            pos_surrogate_derivative: Any = DEFAULT_POS_THRESH["surrogate"],
            neg_surrogate_derivative: Any = DEFAULT_NEG_THRESH["surrogate"],
    ):
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "use_averaging": beta_ema,
            "learnable": learn_beta,
        }

        pos_threshold_setup = {
            "value": pos_threshold,
            "rank": pos_threshold_rank,
            "surrogate": pos_surrogate_derivative,
            "learnable": learn_pos_threshold,
        }

        neg_threshold_setup = {
            "value": neg_threshold,
            "rank": neg_threshold_rank,
            "surrogate": neg_surrogate_derivative,
            "learnable": learn_neg_threshold,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            alpha_setup=None,
            beta_setup=beta_setup,
            gamma_setup=None,
            pos_threshold_setup=pos_threshold_setup,
            neg_threshold_setup=neg_threshold_setup,
            weight_setup=None,
            bias_setup=None,
        )
