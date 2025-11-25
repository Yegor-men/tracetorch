from typing import Union, Literal
import torch
from torch import nn
from ._leaky_integrator import LeakyIntegrator
from .. import functional as tt_functional


class SLIF(LeakyIntegrator):
	def __init__(
			self,
			num_neurons: int,
			alpha: Union[float, torch.Tensor] = 0.5,
			beta: Union[float, torch.Tensor] = 0.9,
			threshold: Union[float, torch.Tensor] = 1.0,
			dim: int = -1,
			alpha_rank: Literal[0, 1] = 1,
			beta_rank: Literal[0, 1] = 1,
			threshold_rank: Literal[0, 1] = 1,
			learn_alpha: bool = True,
			learn_beta: bool = True,
			learn_threshold: bool = True,

			surrogate_derivative=tt_functional.atan_surrogate(2.0)
	):
		alpha_setup = {
			"value": alpha,
			"rank": alpha_rank,
			"use_averaging": False,
			"learnable": learn_alpha,
		}

		beta_setup = {
			"value": beta,
			"rank": beta_rank,
			"use_averaging": False,
			"learnable": learn_beta,
		}

		pos_threshold_setup = {
			"value": threshold,
			"rank": threshold_rank,
			"surrogate": surrogate_derivative,
			"learnable": learn_threshold,
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
