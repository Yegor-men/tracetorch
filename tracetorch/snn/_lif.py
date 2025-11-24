import torch
from torch import nn
from .. import functional as tt_functional
from ._ttmodule import TTModule


class LIF(TTModule):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.9,
			threshold: float = 1.0,
			dim: int = -1,
			beta_rank: int = 1,
			threshold_rank: int = 1,
			learn_beta: bool = True,
			learn_threshold: bool = True,
			surrogate_function=tt_functional.atan_surrogate(2.0),
	):
		super().__init__()
		self.num_neurons = int(num_neurons)
		self.dim = int(dim)
		self.beta_rank = int(beta_rank)
		self.threshold_rank = int(threshold_rank)
		self.learn_beta = bool(learn_beta)
		self.learn_threshold = bool(learn_threshold)
		self.surrogate_function = surrogate_function

		with torch.no_grad():
			if isinstance(beta, torch.Tensor):  # user provided their own beta tensor
				if beta.ndim == 0:  # scalar
					self.beta_rank = 0
				elif beta.ndim == 1:
					assert (beta.numel() == num_neurons), "beta must have num_neurons number of elements"
					self.beta_rank = 1
				else:
					raise ValueError(f"rank (.ndim) of provided beta is not 0 (scalar) or 1 (vector)")
				beta_tensor = tt_functional.sigmoid_inverse(beta)
			else:  # user wants beta tensor generated
				beta = float(beta)
				if self.beta_rank == 0:
					beta_tensor = tt_functional.sigmoid_inverse(torch.tensor(beta))
				elif self.beta_rank == 1:
					beta_tensor = tt_functional.sigmoid_inverse(torch.full([self.num_neurons], beta))
				else:
					raise ValueError("beta_rank is not 0 (scalar) or 1 (vector)")

			if isinstance(threshold, torch.Tensor):  # user provided their own threshold tensor
				if threshold.ndim == 0:  # scalar
					self.threshold_rank = 0
				elif threshold.ndim == 1:  # vector
					assert (threshold.numel() == num_neurons), "threshold must have num_neurons number of elements"
				else:
					raise ValueError(f"rank (.ndim) of provided threshold is not 0 (scalar) or 1 (vector)")
				threshold_tensor = tt_functional.softplus_inverse(threshold)
			else:
				threshold = float(threshold)
				if self.threshold_rank == 0:
					threshold_tensor = tt_functional.softplus_inverse(torch.tensor(threshold))
				elif self.threshold_rank == 1:
					threshold_tensor = tt_functional.softplus_inverse(torch.full([self.num_neurons], threshold))
				else:
					raise ValueError("threshold_rank is not 0 (scalar) or 1 (vector)")

		def register_tensor(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor.detach().clone()))
			else:
				self.register_buffer(name, tensor.detach().clone())

		register_tensor("raw_beta", beta_tensor, learn_beta)
		register_tensor("raw_threshold", threshold_tensor, learn_threshold)

		self.zero_states()

	@property
	def beta(self):
		return nn.functional.sigmoid(self.raw_beta)

	@property
	def threshold(self):
		return nn.functional.softplus(self.raw_threshold)

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

		moved_mem = moved_mem * self.beta + moved_x
		moved_out_spikes = self.surrogate_function(moved_mem - self.threshold)
		moved_mem = moved_mem - moved_out_spikes * self.threshold

		self.mem = moved_mem.movedim(-1, self.dim)
		out_spikes = moved_out_spikes.movedim(-1, self.dim)

		return out_spikes
