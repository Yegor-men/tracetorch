import torch
from torch import nn


class Sequential(nn.Module):
	def __init__(self, *layers):
		super().__init__()
		self.layers = nn.ModuleList(layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, ls: torch.Tensor) -> None:
		for layer in reversed(self.layers):
			ls = layer.backward(ls)
		return ls

	def zero_states(self):
		for layer in self.layers:
			layer.zero_states()

	def clear_grad(self):
		for param in self.parameters():
			param.grad = None

	def get_learnable_parameters(self):
		return [p for layer in self.layers for p in layer.get_learnable_parameters()]
