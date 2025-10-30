import torch
from torch import nn
from .. import functional


class Synaptic(nn.Module):
	def __init__(
			self,
			num_neurons: int,
			alpha: float = 0.5,
			beta: float = 0.5,
			threshold: float = 1.0,
			view_tuple: tuple[int, ...] = (-1,),
			surrogate_function=functional.atan_surrogate(2.0),
			learn_alpha: bool = True,
			learn_beta: bool = True,
			learn_threshold: bool = True,
	):
		super().__init__()
		self.out_features = int(num_neurons)
		self.surrogate_function = surrogate_function
		self.view_tuple = view_tuple

		with torch.no_grad():
			alpha = functional.sigmoid_inverse(torch.ones(num_neurons) * alpha)
			beta = functional.sigmoid_inverse(torch.ones(num_neurons) * beta)
			threshold = functional.softplus_inverse(torch.ones(num_neurons) * threshold)

		def _register(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor))
			else:
				self.register_buffer(name, tensor)

		for (n, t, l) in [
			("alpha", alpha, learn_alpha),
			("beta", beta, learn_beta),
			("threshold", threshold, learn_threshold)
		]:
			_register(n, t, l)

		self.zero_states()

	def zero_states(self):
		self.syn = None
		self.mem = None

	def detach_states(self):
		self.syn = self.syn.detach()
		self.mem = self.mem.detach()

	def forward(self, x):
		if self.syn is None:
			self.syn = torch.zeros_like(x)
		if self.mem is None:
			self.mem = torch.zeros_like(x)

		alpha = nn.functional.sigmoid(self.alpha).view(self.view_tuple)
		beta = nn.functional.sigmoid(self.beta).view(self.view_tuple)
		threshold = nn.functional.softplus(self.threshold).view(self.view_tuple)

		self.syn = self.syn * alpha + x
		self.mem = self.mem * beta + self.syn
		out_spikes = self.surrogate_function(self.mem - threshold)
		self.mem = self.mem - out_spikes * threshold

		return out_spikes
