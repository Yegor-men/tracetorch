import torch
from torch import nn
from .. import functional as tt_functional
from ._ttmodule import TTModule


class RSLIF(TTModule):
	def __init__(
			self,
			num_neurons: int,
			alpha: float = 0.5,
			beta: float = 0.9,
			weight: float = 0.0,
			bias: float = 0.0,
			threshold: float = 1.0,
			dim: int = -1,
			alpha_rank: int = 1,
			beta_rank: int = 1,
			weight_rank: int = 2,
			bias_rank: int = 1,
			threshold_rank: int = 1,
			learn_alpha: bool = True,
			learn_beta: bool = True,
			learn_weight: bool = True,
			learn_bias: bool = True,
			learn_threshold: bool = True,
			surrogate_function=tt_functional.atan_surrogate(2.0),
	):
		super().__init__()
		self.num_neurons = int(num_neurons)
		self.dim = int(dim)
		self.alpha_rank = int(alpha_rank)
		self.beta_rank = int(beta_rank)
		self.weight_rank = int(weight_rank)
		self.bias_rank = int(bias_rank)
		self.threshold_rank = int(threshold_rank)
		self.learn_beta = bool(learn_beta)
		self.learn_weight = bool(learn_weight)
		self.learn_alpha = bool(learn_alpha)
		self.learn_bias = bool(learn_bias)
		self.learn_threshold = bool(learn_threshold)
		self.surrogate_function = surrogate_function

		with torch.no_grad():
			if isinstance(alpha, torch.Tensor):  # user provided their own alpha tensor
				if alpha.ndim == 0:  # scalar
					pass
				elif alpha.ndim == 1:
					assert (alpha.numel() == num_neurons), "alpha must have num_neurons number of elements"
				else:
					raise ValueError(f"rank (.ndim) of provided alpha is not 0 (scalar) or 1 (vector)")
				alpha_tensor = tt_functional.sigmoid_inverse(alpha)
				self.alpha_rank = alpha_tensor.ndim
			else:  # user wants alpha tensor generated
				alpha = float(alpha)
				if self.alpha_rank == 0:
					alpha_tensor = tt_functional.sigmoid_inverse(torch.tensor(alpha))
				elif self.alpha_rank == 1:
					alpha_tensor = tt_functional.sigmoid_inverse(torch.full([self.num_neurons], alpha))
				else:
					raise ValueError("alpha_rank is not 0 (scalar) or 1 (vector)")

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

			if isinstance(weight, torch.Tensor):  # user provided their own weight tensor
				if weight.ndim == 0:  # scalar
					pass
				elif weight.ndim == 1:  # vector
					assert (weight.numel() == num_neurons), "weight must have num_neurons number of elements"
				elif weight.ndim == 2:  # matrix
					assert (weight[0].numel() == weight[1].numel() == num_neurons), "weight must be square matrix"
				else:
					raise ValueError(f"rank (.ndim) of provided weight is not 0 (scalar) or 1 (vector) or 2 (matrix)")
				weight_tensor = weight
				self.weight_rank = weight_tensor.ndim
			else:
				weight = float(weight)
				if self.weight_rank == 0:
					weight_tensor = torch.tensor(weight)
				elif self.weight_rank == 1:
					weight_tensor = torch.full([self.num_neurons], weight)
				elif self.weight_rank == 2:
					weight_tensor = torch.full([self.num_neurons, self.num_neurons], weight)
				else:
					raise ValueError("weight_rank is not 0 (scalar) or 1 (vector) or 2 (matrix)")

			if isinstance(bias, torch.Tensor):  # user provided their own bias tensor
				if bias.ndim == 0:  # scalar
					pass
				elif bias.ndim == 1:  # vector
					assert (bias.numel() == num_neurons), "bias must have num_neurons number of elements"
				else:
					raise ValueError(f"rank (.ndim) of provided bias is not 0 (scalar) or 1 (vector)")
				bias_tensor = bias
				self.bias_rank = bias_tensor.ndim
			else:
				bias = float(bias)
				if self.bias_rank == 0:
					bias_tensor = torch.tensor(bias)
				elif self.bias_rank == 1:
					bias_tensor = torch.full([self.num_neurons], bias)
				else:
					raise ValueError("bias_rank is not 0 (scalar) or 1 (vector)")

			if isinstance(threshold, torch.Tensor):  # user provided their own threshold tensor
				if threshold.ndim == 0:  # scalar
					pass
				elif threshold.ndim == 1:  # vector
					assert (threshold.numel() == num_neurons), "threshold must have num_neurons number of elements"
				else:
					raise ValueError(f"rank (.ndim) of provided threshold is not 0 (scalar) or 1 (vector)")
				threshold_tensor = tt_functional.softplus_inverse(threshold)
				self.threshold_rank = threshold_tensor.ndim
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

		register_tensor("raw_alpha", alpha_tensor, self.learn_alpha)
		register_tensor("raw_beta", beta_tensor, self.learn_beta)
		register_tensor("raw_weight", weight_tensor, self.learn_weight)
		register_tensor("raw_bias", bias_tensor, self.learn_bias)
		register_tensor("raw_threshold", threshold_tensor, self.learn_threshold)

		self.zero_states()

	@property
	def alpha(self):
		return nn.functional.sigmoid(self.raw_alpha)

	@property
	def beta(self):
		return nn.functional.sigmoid(self.raw_beta)

	@property
	def weight(self):
		return self.raw_weight

	@property
	def bias(self):
		return self.raw_bias

	@property
	def threshold(self):
		return nn.functional.softplus(self.raw_threshold)

	def zero_states(self):
		self.syn = None
		self.mem = None
		self.prev_output = None

	def detach_states(self):
		if self.syn is not None:
			self.syn = self.syn.detach()
		if self.mem is not None:
			self.mem = self.mem.detach()
		if self.prev_output is not None:
			self.prev_output = self.prev_output.detach()

	def forward(self, x):
		if self.syn is None:
			self.syn = torch.zeros_like(x)
		if self.mem is None:
			self.mem = torch.zeros_like(x)
		if self.prev_output is None:
			self.prev_output = torch.zeros_like(x)

		moved_x = x.movedim(self.dim, -1)
		moved_syn = self.syn.movedim(self.dim, -1)
		moved_mem = self.mem.movedim(self.dim, -1)
		moved_prev_output = self.prev_output.movedim(self.dim, -1)

		moved_syn = moved_syn * self.alpha + moved_x

		if self.weight_rank == 2:
			moved_syn = moved_syn + moved_prev_output @ self.weight + self.bias
		else:
			moved_syn = moved_syn + moved_prev_output * self.weight + self.bias

		moved_mem = moved_mem * self.beta + moved_syn
		
		moved_out_spikes = self.surrogate_function(moved_mem - self.threshold)
		moved_mem = moved_mem - moved_out_spikes * self.threshold

		out_spikes = moved_out_spikes.movedim(-1, self.dim)
		self.syn = moved_syn.movedim(-1, self.dim)
		self.mem = moved_mem.movedim(-1, self.dim)
		self.prev_output = out_spikes

		return out_spikes
