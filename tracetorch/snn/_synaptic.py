import torch
from torch import nn
from tracetorch import functional


class Synaptic(nn.Module):
	def __init__(
			self,
			num_neurons: int,
			alpha: float = 0.5,
			beta: float = 0.5,
			threshold: float = 1.0,
			return_mem: bool = True,
			view_tuple: tuple = (-1),
			surrogate_function=functional.atan_surrogate(2.0),
	):
		super().__init__()
		self.out_features = int(num_neurons)
		self.surrogate_function = surrogate_function
		self.return_mem = return_mem
		self.view_tuple = view_tuple

		with torch.no_grad():
			alpha = functional.sigmoid_inverse(torch.ones(num_neurons) * alpha)
			beta = functional.sigmoid_inverse(torch.ones(num_neurons) * beta)
			threshold = functional.softplus_inverse(torch.ones(num_neurons) * threshold)

		self.alpha = nn.Parameter(alpha)
		self.beta = nn.Parameter(beta)
		self.threshold = nn.Parameter(threshold)

		self.zero_states()

	def zero_states(self):
		self.syn = None
		self.mem = None

	def detach_states(self):
		self.syn = self.syn.detach()
		self.mem = self.mem.detach()

	def param_to_dim(self, parameter):
		return parameter.view(self.view_tuple)

	def forward(self, x):
		if self.syn is None:
			self.syn = torch.zeros_like(x)
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		alpha = self.param_to_dim(nn.functional.sigmoid(self.alpha))
		beta = self.param_to_dim(nn.functional.sigmoid(self.beta))
		threshold = self.param_to_dim(nn.functional.softplus(self.threshold))

		self.syn = self.syn * alpha + x
		self.mem = self.mem * beta + self.syn

		if self.return_mem:
			return self.mem
		else:
			out_spikes = self.surrogate_function(self.mem - threshold)
			self.mem = self.mem - out_spikes * threshold
			return out_spikes
