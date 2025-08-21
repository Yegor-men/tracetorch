import torch
from torch import nn


class LIS(nn.Module):
	def __init__(
			self,
			num_in: int,
			num_out: int,
			pre_decay: float = 0.9,
	):
		super().__init__()
		self.num_in = int(num_in)
		self.num_out = int(num_out)

		self.weight = nn.Parameter(torch.randn(num_out, num_in))
		self.bias = nn.Parameter(torch.zeros(num_out))

		self.register_buffer("pre_decay", torch.tensor(pre_decay))

		self.register_buffer("pre_trace", torch.zeros(num_in))

		self.register_buffer("normalized_pre_trace", torch.zeros(num_in))

	@torch.no_grad()
	def zero_states(self):
		self.pre_trace.zero_()
		self.normalized_pre_trace.zero_()

	@torch.no_grad()
	def forward(self, in_spikes):
		self.pre_trace.mul_(self.pre_decay).add_(in_spikes)

		self.normalized_pre_trace.zero_().add_(self.pre_trace * (1 - self.pre_decay))
		logits = torch.einsum("oi, i -> o", self.weight, self.normalized_pre_trace) + self.bias
		probability_distribution = nn.functional.softmax(logits, dim=-1)

		return probability_distribution

	@torch.no_grad()
	def backward(self, learning_signal):
		weight_grad = torch.einsum("o, i -> oi", learning_signal, self.normalized_pre_trace)

		if self.weight.grad is None:
			self.weight.grad = weight_grad.clone()
		else:
			self.weight.grad += weight_grad

		if self.bias.grad is None:
			self.bias.grad = learning_signal.clone()
		else:
			self.bias.grad += learning_signal

		passed_ls = torch.einsum("oi, o -> i", self.weight, learning_signal) * (1 - self.pre_decay)

		return passed_ls
