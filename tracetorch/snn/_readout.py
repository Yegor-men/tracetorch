import torch
from torch import nn
from .. import functional


class Readout(nn.Module):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.99,
			view_tuple: tuple = (-1),
			learn_beta: bool = True,
			beta_is_scalar: bool = True,
	):
		super().__init__()
		self.view_tuple = view_tuple
		self.beta_is_scalar = beta_is_scalar
		self.num_neurons = num_neurons

		num_neurons = num_neurons if not beta_is_scalar else 1
		with torch.no_grad():
			beta = functional.sigmoid_inverse(torch.ones(num_neurons) * beta)

		def _register(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor))
			else:
				self.register_buffer(name, tensor)

		for (n, t, l) in [
			("beta", beta, learn_beta),
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

		beta = nn.functional.sigmoid(self.beta)
		if self.beta_is_scalar:
			beta = beta * torch.ones(self.num_neurons).to(x)
		beta = beta.view(self.view_tuple)

		self.mem = self.mem * beta + x * (1 - beta)

		return self.mem
