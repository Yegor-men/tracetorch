import torch
from torch import nn
from .. import functional
from ._base_module import BaseModule


class LIF(BaseModule):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.9,
			threshold: float = 1.0,
			view_tuple: tuple[int, ...] = (-1,),
			surrogate_function=functional.atan_surrogate(2.0),
			learn_beta: bool = True,
			learn_threshold: bool = True,
			beta_is_vector: bool = True,
			threshold_is_vector: bool = True,
	):
		super().__init__()
		self.out_features = int(num_neurons)
		self.surrogate_function = surrogate_function
		self.view_tuple = view_tuple

		with torch.no_grad():
			if isinstance(beta, torch.Tensor):
				beta_scalar = torch.tensor(1.)
				beta_vector = functional.sigmoid_inverse(beta.clone().detach())
			else:
				beta_scalar = functional.sigmoid_inverse(torch.tensor(beta))
				beta_vector = torch.ones(num_neurons)

			if isinstance(threshold, torch.Tensor):
				threshold_scalar = torch.tensor(1.)
				threshold_vector = functional.softplus_inverse(beta.clone().detach())
			else:
				threshold_scalar = functional.softplus_inverse(torch.tensor(threshold))
				threshold_vector = torch.ones(num_neurons)

		for (n, t, l) in [
			("beta_scalar", beta_scalar, (learn_beta and not beta_is_vector)),
			("beta_vector", beta_vector, (learn_beta and beta_is_vector)),
			("threshold_scalar", threshold_scalar, (learn_threshold and not threshold_is_vector)),
			("threshold_vector", threshold_vector, (learn_threshold and threshold_is_vector))
		]:
			self._register_tensor(n, t, l)

		self.zero_states()

	@property
	def beta(self):
		return nn.functional.sigmoid(self.beta_vector * self.beta_scalar)

	@property
	def threshold(self):
		return nn.functional.softplus(self.threshold_vector * self.threshold_scalar)

	def zero_states(self):
		self.mem = None

	def detach_states(self):
		self.mem = self.mem.detach()

	def forward(self, x):
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		beta = self.beta.view(self.view_tuple)
		threshold = self.threshold.view(self.view_tuple)

		self.mem = self.mem * beta + x
		out_spikes = self.surrogate_function(self.mem - threshold)
		self.mem = self.mem - out_spikes * threshold

		return out_spikes
