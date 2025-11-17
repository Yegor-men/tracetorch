import torch
from torch import nn
from .. import functional
from ._base_module import BaseModule


class Readout(BaseModule):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.9,
			view_tuple: tuple[int, ...] = (-1,),
			learn_beta: bool = True,
			beta_is_vector: bool = False,
	):
		super().__init__()
		self.view_tuple = view_tuple
		self.num_neurons = num_neurons

		with torch.no_grad():
			beta_scalar = functional.sigmoid_inverse(torch.tensor(beta))
			beta_vector = torch.ones(num_neurons)

		for (n, t, l) in [
			("beta_scalar", beta_scalar, learn_beta),
			("beta_vector", beta_vector, (learn_beta and beta_is_vector)),
		]:
			self._register_tensor(n, t, l)

		self.zero_states()

	@property
	def beta(self):
		return nn.functional.sigmoid(self.beta_scalar * self.beta_vector)

	def zero_states(self):
		self.mem = None

	def detach_states(self):
		self.mem = self.mem.detach()

	def forward(self, x):
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		beta = nn.functional.sigmoid(self.beta_scalar * self.beta_vector)
		beta = beta.view(self.view_tuple)

		self.mem = self.mem * beta + x * (1 - beta)

		return self.mem
