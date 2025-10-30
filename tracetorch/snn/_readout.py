import torch
from torch import nn
from .. import functional


class Readout(nn.Module):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.99,
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

		def _register(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor))
			else:
				self.register_buffer(name, tensor)

		for (n, t, l) in [
			("beta_scalar", beta_scalar, learn_beta),
			("beta_vector", beta_vector, (learn_beta and beta_is_vector)),
		]:
			_register(n, t, l)

		self.zero_states()

	def zero_states(self):
		self.mem = None

	def detach_states(self):
		self.mem = self.mem.detach()

	def forward(self, x):
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		beta = nn.functional.sigmoid(self.beta_vector * self.beta_scalar)
		beta = beta.view(self.view_tuple)

		self.mem = self.mem * beta + x * (1 - beta)

		return self.mem
