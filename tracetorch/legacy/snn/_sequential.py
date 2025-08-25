import torch
from torch import nn


class Sequential(nn.Module):
	def __init__(self, *layers):
		super().__init__()
		self.layers = nn.ModuleList(layers)
		self.num_in = int(self.layers[0].num_in)
		self.num_out = int(self.layers[-1].num_out)

	@torch.no_grad()
	def decay_grad(self, decay):
		for p in self.parameters():
			if p is not None:
				p.mul_(decay)

	@torch.no_grad()
	def optim_trace_step(self, optimizer: torch.optim, decay):
		clones = [p.grad.detach().clone() if p is not None else None for p in self.parameters()]
		for p in self.parameters():
			p.grad.mul_(1 - decay)
		optimizer.step()
		for p, c in zip(self.parameters(), clones):
			p.grad.copy_(c)
		del clones

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, ls: torch.Tensor) -> None:
		for layer in reversed(self.layers):
			ls = layer.backward(ls)

	def zero_states(self):
		for layer in self.layers:
			layer.zero_states()

	def get_learnable_parameters(self):
		return [p for layer in self.layers for p in layer.get_learnable_parameters()]