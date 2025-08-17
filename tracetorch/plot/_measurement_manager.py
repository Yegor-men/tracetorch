import torch
from .. import plot


class MeasurementManager:
	"""
	Manages measurements over time like loss or accuracy
	"""

	def __init__(
			self,
			title: str,
			decay: list = [0., 0.9, 0.99, 0.999]
	):
		self.title = title
		self.decay = torch.tensor(decay).float()
		self.trace = torch.zeros_like(self.decay)
		self.measurement = []

	def append(self, value):
		with torch.no_grad():
			if isinstance(value, torch.Tensor):
				value = value.item()
			self.trace *= self.decay
			self.trace += value * torch.ones_like(self.trace)
			avg_input = self.trace * (1 - self.decay)
			self.measurement.append(avg_input)

	def plot(self, title: str = None):
		plot_title = title if title is not None else self.title
		rounded_list = [round(x, 4) for x in self.decay.tolist()]
		plot.line_graph(self.measurement, title=plot_title, label=rounded_list)
