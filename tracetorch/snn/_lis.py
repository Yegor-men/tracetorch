import torch
from torch import nn
from .. import functional


class LIS(nn.Module):
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float = 0.9,
			threshold: float = 1,
			in_trace_decay: float = 0.9,
			init_weight_scaling: float = 0.1,
			learn_weight: bool = True,
			learn_mem_decay: bool = True,
			learn_in_trace_decay: bool = True,
	):
		super().__init__()
		self.learn_weight = learn_weight
		self.learn_mem_decay = learn_mem_decay
		self.learn_in_trace_decay = learn_in_trace_decay

		self.weight = nn.Parameter(torch.randn(num_out, num_in) * init_weight_scaling)
		self.mem_decay = nn.Parameter(functional.sigmoid_inverse(torch.full((num_out,), mem_decay)))
		self.threshold = nn.Parameter(functional.softplus_inverse(torch.full((num_out,), threshold)))
		self.in_trace_decay = nn.Parameter(torch.full((num_in,), in_trace_decay))

		self.register_buffer("mem", torch.zeros(num_out))
		self.register_buffer("in_trace", torch.zeros(num_in))

	def get_learnable_parameters(self):
		learnable_parameters = [
			tensor
			for tensor, learnable in [
				(self.weight, self.learn_weight),
				(self.mem_decay, self.learn_mem_decay),
				(self.in_trace_decay, self.learn_in_trace_decay)
			] if learnable
		]

		return learnable_parameters

	def clear_grad(self):
		for param in self.parameters():
			param.grad = None

	def zero_states(self):
		self.mem.zero_()
		self.in_trace.zero_()

	def forward(self, in_spikes):
		with torch.no_grad():
			in_trace_decay = nn.functional.sigmoid(self.in_trace_decay)
			self.in_trace.mul_(in_trace_decay).add_(in_spikes)
			synaptic_current = torch.einsum("i, oi -> o", in_spikes, self.weight)
			mem_decay = nn.functional.sigmoid(self.mem_decay)
			self.mem.mul_(mem_decay).add_(synaptic_current)
			probability_dist = torch.nn.functional.softmax(self.mem, dim=-1)
			return probability_dist

	def backward(self, learning_signal):
		in_trace_decay = nn.functional.sigmoid(self.in_trace_decay)
		average_input = self.in_trace * (1 - in_trace_decay)
		average_input.retain_grad()
		i = torch.einsum("i, oi -> o", average_input, self.weight)
		d = nn.functional.sigmoid(self.mem_decay)

		stabilized_mem_level = i / (1 - d + 1e-6)
		frequency = torch.nn.functional.softmax(stabilized_mem_level, dim=-1)

		frequency.backward(learning_signal.detach())
		passed_learning_signal = average_input.grad.detach().clone()
		average_input.grad = None
		del average_input
		return passed_learning_signal