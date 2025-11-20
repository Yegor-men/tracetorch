import torch
from torch import nn
from .. import functional
from ._ttmodule import TTModule


class Readout(TTModule):
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
			if isinstance(beta, torch.Tensor):
				# if user provided a custom beta
				if beta.ndim == 0:  # beta is scalar
					beta_scalar = functional.sigmoid_inverse(beta)
					beta_vector = torch.ones(num_neurons)
					beta_is_vector = False  # override in case it's wrong
				else:
					assert (beta.ndim == 1) and (beta.numel() == num_neurons)  # beta must be a vector
					beta_scalar = torch.tensor(1.)
					beta_vector = functional.sigmoid_inverse(beta)
					beta_is_vector = True  # override in case it's wrong
			else:
				beta = float(beta)
				assert 0.0 < beta < 1.0  # beta must be in (0,1)
				if beta_is_vector:  # want beta to be a vector
					beta_scalar = torch.tensor(1.)
					beta_vector = functional.sigmoid_inverse(torch.full((num_neurons,), beta))
				else:  # want beta to be a scalar
					beta_scalar = functional.sigmoid_inverse(torch.tensor(beta))
					beta_vector = torch.ones(num_neurons)

		def register_tensor(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor.detach().clone()))
			else:
				self.register_buffer(name, tensor.detach().clone())

		for (n, t, l) in [
			("beta_scalar", beta_scalar, (learn_beta and not beta_is_vector)),
			("beta_vector", beta_vector, (learn_beta and beta_is_vector)),
		]:
			register_tensor(n, t, l)

		self.zero_states()

	@property
	def beta(self):
		return nn.functional.sigmoid(self.beta_scalar * self.beta_vector)

	def zero_states(self):
		self.mem = None

	def detach_states(self):
		if self.mem is not None:
			self.mem = self.mem.detach()

	def forward(self, x):
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		beta = self.beta.view(self.view_tuple)

		self.mem = self.mem * beta + x * (1 - beta)

		return self.mem
