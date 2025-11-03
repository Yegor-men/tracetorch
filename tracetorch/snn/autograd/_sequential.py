import torch
from torch import nn


class Sequential(nn.Module):
	def __init__(self, *layers):
		super().__init__()

		self.layers = nn.ModuleList(layers)

	def _call_if_present(self, layer, name: str, *args, **kwargs):
		"""Call method `name` on each immediate child if it exists and is callable."""
		fn = getattr(layer, name, None)
		if callable(fn):
			fn(*args, **kwargs)

	def zero_states(self):
		for layer in self.layers:
			self._call_if_present(layer, "zero_states")

	def detach_states(self):
		for layer in self.layers:
			self._call_if_present(layer, "detach_states")

	def get_attr_list(self, *attr_names):
		return [getattr(l, n) for l in self.layers for n in attr_names if hasattr(l, n)]

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
