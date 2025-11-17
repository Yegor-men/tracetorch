import torch
from torch import nn


class BaseModule(nn.Module):
	def __init__(self):
		super().__init__()

	def _register_tensor(self, name: str, tensor: torch.Tensor, learn: bool):
		if learn:
			setattr(self, name, nn.Parameter(tensor))
		else:
			self.register_buffer(name, tensor)
