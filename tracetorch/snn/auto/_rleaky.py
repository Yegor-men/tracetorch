import torch
from torch import nn
from ... import functional


class RLeaky(nn.Module):
	def __init__(
			self,
			num_neurons: int,
			beta: float = 0.9,
			threshold: float = 1.0,
			v: float = 1.0,
			view_tuple: tuple[int, ...] = (-1,),
			surrogate_function=functional.atan_surrogate(2.0),
			learn_beta: bool = True,
			learn_threshold: bool = True,
			learn_v: bool = True,
			v_is_vector: bool = True,
	):
		super().__init__()
		self.out_features = int(num_neurons)
		self.surrogate_function = surrogate_function
		self.view_tuple = view_tuple

		with torch.no_grad():
			beta = functional.sigmoid_inverse(torch.ones(num_neurons) * beta)
			threshold = functional.softplus_inverse(torch.ones(num_neurons) * threshold)
			v_scalar = torch.tensor(v)
			v_vector = torch.ones(num_neurons)

		def _register(name: str, tensor: torch.Tensor, learn: bool):
			if learn:
				setattr(self, name, nn.Parameter(tensor))
			else:
				self.register_buffer(name, tensor)

		for (n, t, l) in [
			("beta", beta, learn_beta),
			("threshold", threshold, learn_threshold),
			("v_scalar", v_scalar, learn_v),
			("v_vector", v_vector, (learn_v and v_is_vector)),
		]:
			_register(n, t, l)

		self.zero_states()

	def zero_states(self):
		self.mem = None
		self.prev_out = None

	def detach_states(self):
		self.mem = self.mem.detach()
		self.prev_out = self.prev_out.detach()

	def forward(self, x):
		if self.mem is None:
			self.mem = torch.zeros_like(x)
		if self.prev_out is None:
			self.prev_out = torch.zeros_like(x)

		beta = nn.functional.sigmoid(self.beta).view(self.view_tuple)
		v = (self.v_vector * self.v_scalar).view(self.view_tuple)
		threshold = nn.functional.softplus(self.threshold).view(self.view_tuple)

		self.mem = self.mem * beta + self.prev_out * v + x
		out_spikes = self.surrogate_function(self.mem - threshold)
		self.mem = self.mem - out_spikes * threshold
		self.prev_out = out_spikes

		return out_spikes
