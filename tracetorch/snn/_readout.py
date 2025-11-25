from typing import Union, Literal
import torch
from torch import nn
from ._leaky_integrator import LeakyIntegrator
from .. import functional as tt_functional


class Readout(LeakyIntegrator):
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
			"use_averaging": True,
			"learnable": learn_beta,
		}

		super().__init__(
			num_neurons=num_neurons,
			dim=dim,
			alpha_setup=None,
			beta_setup=beta_setup,
			gamma_setup=None,
			pos_threshold_setup=None,
			neg_threshold_setup=None,
			weight_setup=None,
			bias_setup=None,
		)
