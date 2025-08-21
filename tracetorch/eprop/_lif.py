import torch
from torch import nn
from ..functional import sigmoid_inverse
from ..functional import softplus_inverse
from ..functional import sigmoid_derivative
import math


class LIF(nn.Module):
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float = 0.9,
			threshold: float = 1,
			pre_decay: float = 0.9,
			post_decay: float = 0.9,
	):
		super().__init__()
		self.num_in = int(num_in)
		self.num_out = int(num_out)

		t, i = threshold, num_in
		self.weight = nn.Parameter(torch.normal(mean=t / i, std=t / math.sqrt(i), size=(num_out, num_in)))
		self.mem_decay = nn.Parameter(sigmoid_inverse(torch.full((num_out,), mem_decay)))
		self.threshold = nn.Parameter(softplus_inverse(torch.full((num_out,), threshold)))

		self.register_buffer("pre_decay", torch.tensor(pre_decay))
		self.register_buffer("post_decay", torch.tensor(post_decay))

		self.register_buffer("mem", torch.zeros(num_out))
		self.register_buffer("pre_trace", torch.zeros(num_in))
		self.register_buffer("post_trace", torch.zeros(num_out))

		self.register_buffer("last_surrogate", torch.zeros(num_out))

	@torch.no_grad()
	def zero_states(self):
		self.mem.zero_()
		self.pre_trace.zero_()
		self.post_trace.zero_()
		self.last_surrogate.zero_()

	@torch.no_grad()
	def forward(self, in_spikes: torch.Tensor):
		self.pre_trace.mul_(self.pre_decay).add_(in_spikes)

		synaptic_current = torch.einsum("oi, i -> o", self.weight, in_spikes)
		mem_decay = nn.functional.sigmoid(self.mem_decay)
		self.mem.mul_(mem_decay).add_(synaptic_current)
		threshold = nn.functional.softplus(self.threshold)
		out_spikes = (self.mem >= threshold).float()

		self.last_surrogate.zero_().copy_(sigmoid_derivative(self.mem - threshold))
		self.post_trace.mul_(self.post_decay).add_(self.last_surrogate)

		self.mem.sub_(out_spikes * threshold)

		return out_spikes

	@torch.no_grad()
	def backward(self, learning_signal: torch.Tensor):
		normalized_pre = self.pre_trace * (1 - self.pre_decay)
		normalized_post = self.post_trace * (1 - self.post_decay)
		e_trace = torch.einsum("i, o -> oi", normalized_pre, normalized_post)
		weight_grad = torch.einsum("oi, o -> oi", e_trace, learning_signal)
		passed_ls = torch.einsum("oi, o -> i", self.weight, learning_signal * self.last_surrogate)
		if self.weight.grad is None:
			self.weight.grad = weight_grad.clone()
		else:
			self.weight.grad += weight_grad
		return passed_ls
