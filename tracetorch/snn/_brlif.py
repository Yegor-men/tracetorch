from ._leaky_integrator import LeakyIntegrator
# from ._leaky_integrator import DEFAULT_ALPHA
from ._leaky_integrator import DEFAULT_BETA
from ._leaky_integrator import DEFAULT_GAMMA
from ._leaky_integrator import DEFAULT_POS_THRESH
from ._leaky_integrator import DEFAULT_NEG_THRESH
from ._leaky_integrator import DEFAULT_POS_SCALE
from ._leaky_integrator import DEFAULT_NEG_SCALE
from ._leaky_integrator import DEFAULT_WEIGHT
from ._leaky_integrator import DEFAULT_BIAS

from typing import Union, Literal, Any
import torch


class BRLIF(LeakyIntegrator):
	def __init__(
			self,
			num_neurons: int,
			beta: Union[float, torch.Tensor] = DEFAULT_BETA["value"],
			gamma: Union[float, torch.Tensor] = DEFAULT_GAMMA["value"],
			pos_threshold: Union[float, torch.Tensor] = DEFAULT_POS_THRESH["value"],
			neg_threshold: Union[float, torch.Tensor] = DEFAULT_NEG_THRESH["value"],
			pos_scale: Union[float, torch.Tensor] = DEFAULT_POS_SCALE["value"],
			neg_scale: Union[float, torch.Tensor] = DEFAULT_NEG_SCALE["value"],
			weight: Union[float, torch.Tensor] = DEFAULT_WEIGHT["value"],
			bias: Union[float, torch.Tensor] = DEFAULT_BIAS["value"],
			dim: int = -1,
			beta_rank: Literal[0, 1] = DEFAULT_BETA["rank"],
			gamma_rank: Literal[0, 1] = DEFAULT_GAMMA["rank"],
			pos_threshold_rank: Literal[0, 1] = DEFAULT_POS_THRESH["rank"],
			neg_threshold_rank: Literal[0, 1] = DEFAULT_NEG_THRESH["rank"],
			pos_scale_rank: Literal[0, 1] = DEFAULT_POS_SCALE["rank"],
			neg_scale_rank: Literal[0, 1] = DEFAULT_NEG_SCALE["rank"],
			weight_rank: Literal[0, 1, 2] = DEFAULT_WEIGHT["rank"],
			bias_rank: Literal[0, 1] = DEFAULT_BIAS["rank"],
			beta_ema: bool = DEFAULT_BETA["use_averaging"],
			gamma_ema: bool = DEFAULT_GAMMA["use_averaging"],
			learn_beta: bool = DEFAULT_BETA["learnable"],
			learn_gamma: bool = DEFAULT_GAMMA["learnable"],
			learn_pos_threshold: bool = DEFAULT_POS_THRESH["learnable"],
			learn_neg_threshold: bool = DEFAULT_NEG_THRESH["learnable"],
			learn_pos_scale: bool = DEFAULT_POS_SCALE["learnable"],
			learn_neg_scale: bool = DEFAULT_NEG_SCALE["learnable"],
			learn_weight: bool = DEFAULT_WEIGHT["learnable"],
			learn_bias: bool = DEFAULT_BIAS["learnable"],
			pos_surrogate_derivative: Any = DEFAULT_POS_THRESH["surrogate"],
			neg_surrogate_derivative: Any = DEFAULT_NEG_THRESH["surrogate"],
	):
		beta_setup = {
			"value": beta,
			"rank": beta_rank,
			"use_averaging": beta_ema,
			"learnable": learn_beta,
		}

		gamma_setup = {
			"value": gamma,
			"rank": gamma_rank,
			"use_averaging": gamma_ema,
			"learnable": learn_gamma,
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

		pos_scale_setup = {
			"value": pos_scale,
			"rank": pos_scale_rank,
			"learnable": learn_pos_scale,
		}

		neg_scale_setup = {
			"value": neg_scale,
			"rank": neg_scale_rank,
			"learnable": learn_neg_scale,
		}

		weight_setup = {
			"value": weight,
			"rank": weight_rank,
			"learnable": learn_weight,
		}

		bias_setup = {
			"value": bias,
			"rank": bias_rank,
			"learnable": learn_bias,
		}

		super().__init__(
			num_neurons=num_neurons,
			dim=dim,
			alpha_setup=None,
			beta_setup=beta_setup,
			gamma_setup=gamma_setup,
			pos_threshold_setup=pos_threshold_setup,
			neg_threshold_setup=neg_threshold_setup,
			pos_scale_setup=pos_scale_setup,
			neg_scale_setup=neg_scale_setup,
			weight_setup=weight_setup,
			bias_setup=bias_setup,
		)
