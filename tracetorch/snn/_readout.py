from ._leaky_integrator import LeakyIntegrator
# from ._leaky_integrator import DEFAULT_ALPHA
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


class Readout(LeakyIntegrator):
	def __init__(
			self,
			num_neurons: int,
			beta: Union[float, torch.Tensor] = DEFAULT_BETA["value"],
			dim: int = -1,
			beta_rank: Literal[0, 1] = DEFAULT_BETA["rank"],
			learn_beta: bool = DEFAULT_BETA["learnable"],
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
			beta_setup=beta_setup,
		)
