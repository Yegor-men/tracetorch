import torch
from torch import nn
from .. import functional as tt_functional
from ._ttmodule import TTModule


class Readout(TTModule):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.9,
			dim: int = -1,
			beta_rank: int = 0,
			learn_beta: bool = True,
	):
		super().__init__()
		self.num_neurons = int(num_neurons)
		self.dim = int(dim)
		self.beta_rank = int(beta_rank)
		self.learn_beta = bool(learn_beta)

		with torch.no_grad():
			if isinstance(beta, torch.Tensor):  # user provided their own beta tensor
				if beta.ndim == 0:  # scalar
					pass
				elif beta.ndim == 1:
					assert (beta.numel() == num_neurons), "beta must have num_neurons number of elements"
				else:
					raise ValueError(f"rank (.ndim) of provided beta is not 0 (scalar) or 1 (vector)")
				beta_tensor = tt_functional.sigmoid_inverse(beta)
				self.beta_rank = beta_tensor.ndim
			else:  # user wants beta tensor generated
				beta = float(beta)
				if self.beta_rank == 0:
					beta_tensor = tt_functional.sigmoid_inverse(torch.tensor(beta))
				elif self.beta_rank == 1:
					beta_tensor = tt_functional.sigmoid_inverse(torch.full([self.num_neurons], beta))
				else:
					raise ValueError("beta_rank is not 0 (scalar) or 1 (vector)")

		def register_tensor(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor.detach().clone()))
			else:
				self.register_buffer(name, tensor.detach().clone())

		register_tensor("raw_beta", beta_tensor, self.learn_beta)

		self.zero_states()

	@property
	def beta(self):
		return nn.functional.sigmoid(self.raw_beta)

	def zero_states(self):
		self.mem = None

	def detach_states(self):
		if self.mem is not None:
			self.mem = self.mem.detach()

	def forward(self, x):
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		moved_x = x.movedim(self.dim, -1)
		moved_mem = self.mem.movedim(self.dim, -1)

		moved_mem = moved_mem * self.beta + moved_x * (1 - self.beta)

		self.mem = moved_mem.movedim(-1, self.dim)

		return self.mem
