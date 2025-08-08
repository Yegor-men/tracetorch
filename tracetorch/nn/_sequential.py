import torch


class Sequential:
	def __init__(self, *layers):
		self.layers = list(layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for index, layer in enumerate(self.layers):
			x = layer.forward(x)
		return x

	def backward(self, ls: torch.Tensor) -> None:
		for index, layer in enumerate(reversed(self.layers)):
			ls = layer.backward(ls)

	def zero_states(self):
		for layer in self.layers:
			layer.zero_states()