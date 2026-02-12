from tracetorch.snn.flex._leaky_integrator import LeakyIntegrator
# from ._leaky_integrator import DEFAULT_ALPHA
from tracetorch.snn.flex._leaky_integrator import DEFAULT_BETA
from tracetorch.snn.flex._leaky_integrator import DEFAULT_GAMMA
# from ._leaky_integrator import DEFAULT_POS_THRESH
# from ._leaky_integrator import DEFAULT_NEG_THRESH
# from ._leaky_integrator import DEFAULT_POS_SCALE
# from ._leaky_integrator import DEFAULT_NEG_SCALE
from tracetorch.snn.flex._leaky_integrator import DEFAULT_REC_WEIGHT
from tracetorch.snn.flex._leaky_integrator import DEFAULT_BIAS

from typing import Union, Literal
import torch


class RReadout(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = DEFAULT_BETA["value"],
            gamma: Union[float, torch.Tensor] = DEFAULT_GAMMA["value"],
            rec_weight: Union[float, torch.Tensor] = DEFAULT_REC_WEIGHT["value"],
            bias: Union[float, torch.Tensor] = DEFAULT_BIAS["value"],
            dim: int = -1,
            beta_rank: Literal[0, 1] = DEFAULT_BETA["rank"],
            gamma_rank: Literal[0, 1] = DEFAULT_GAMMA["rank"],
            rec_weight_rank: Literal[0, 1] = DEFAULT_REC_WEIGHT["rank"],
            bias_rank: Literal[0, 1] = DEFAULT_BIAS["rank"],
            ema_gamma: bool = DEFAULT_GAMMA["ema"],
            learn_beta: bool = DEFAULT_BETA["learnable"],
            learn_gamma: bool = DEFAULT_GAMMA["learnable"],
            learn_rec_weight: bool = DEFAULT_REC_WEIGHT["learnable"],
            learn_bias: bool = DEFAULT_BIAS["learnable"],
    ):
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": True,
            "learnable": learn_beta,
        }

        gamma_setup = {
            "value": gamma,
            "rank": gamma_rank,
            "ema": ema_gamma,
            "learnable": learn_gamma,
        }

        rec_weight_setup = {
            "value": rec_weight,
            "rank": rec_weight_rank,
            "learnable": learn_rec_weight,
        }

        bias_setup = {
            "value": bias,
            "rank": bias_rank,
            "learnable": learn_bias,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            beta_pos_setup=beta_setup,
            gamma_pos_setup=gamma_setup,
            rec_weight_setup=rec_weight_setup,
            bias_setup=bias_setup,
        )
