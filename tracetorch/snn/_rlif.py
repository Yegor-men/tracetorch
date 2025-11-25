from typing import Union, Literal
import torch
from torch import nn
from ._leaky_integrator import LeakyIntegrator
from .. import functional as tt_functional


class RLIF(LeakyIntegrator):
	def __init__(
			self,
			num_neurons: int,
			beta: Union[float, torch.Tensor] = 0.9,
			gamma: Union[float, torch.Tensor] = 0.9,
			threshold: Union[float, torch.Tensor] = 1.0,
			weight: Union[float, torch.Tensor] = 0.0,
			bias: Union[float, torch.Tensor] = 0.0,
			dim: int = -1,
			beta_rank: Literal[0, 1] = 1,
			gamma_rank: Literal[0, 1] = 1,
			threshold_rank: Literal[0, 1] = 1,
			weight_rank: Literal[0, 1, 2] = 2,
			bias_rank: Literal[0, 1] = 1,
			learn_beta: bool = True,
			learn_gamma: bool = True,
			learn_threshold: bool = True,
			learn_weight: bool = True,
			learn_bias: bool = True,
			surrogate_derivative=tt_functional.atan_surrogate(2.0)
	):
		beta_setup = {
			"value": beta,
			"rank": beta_rank,
			"use_averaging": False,
			"learnable": learn_beta,
		}

		gamma_setup = {
			"value": gamma,
			"rank": gamma_rank,
			"use_averaging": False,
			"learnable": learn_gamma,
		}

		pos_threshold_setup = {
			"value": threshold,
			"rank": threshold_rank,
			"surrogate": surrogate_derivative,
			"learnable": learn_threshold,
		}

		weight_setup = {
			"value": weight,
			"rank": weight_rank,
			"connect_to": "rec",
			"learnable": learn_weight,
		}

		bias_setup = {
			"value": bias,
			"rank": bias_rank,
			"connect_to": "rec",
			"learnable": learn_bias,
		}

		super().__init__(
			num_neurons=num_neurons,
			dim=dim,
			alpha_setup=None,
			beta_setup=beta_setup,
			gamma_setup=gamma_setup,
			pos_threshold_setup=pos_threshold_setup,
			neg_threshold_setup=None,
			weight_setup=weight_setup,
			bias_setup=bias_setup,
		)
