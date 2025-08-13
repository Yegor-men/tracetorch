import torch
from tracetorch import functional

class Reflect:
	def __init__(
			self,
			num_out: int,
			decay: float = 0.9,
	):
		self.distribution_trace = torch.zeros(num_out)
		self.output_trace = torch.zeros(num_out)
		self.decay = functional.sigmoid_inverse(torch.ones(num_out) * decay)

	def forward(self, distribution, output):
		with torch.no_grad():
			self.distribution_trace = self.distribution_trace * self.decay + distribution
			self.output_trace = self.output_trace * self.decay + distribution