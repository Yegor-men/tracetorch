import torch
from .. import plot


class MeasurementManager:
	"""Track and plot exponentially smoothed measurements.

	``MeasurementManager`` stores raw scalar measurements and several exponential
	moving averages with different decays. It is useful for quick experiment
	tracking of losses, accuracies, or other scalar diagnostics.

	Args:
		title (str): default plot title.
		decay (list, default=[0., 0.9, 0.99, 0.999]): EMA decay values to track.
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
		"""Append a scalar value and update all EMA traces."""
		with torch.no_grad():
			if isinstance(value, torch.Tensor):
				value = value.item()
			self.trace *= self.decay
			self.trace += value * torch.ones_like(self.trace)
			avg_input = self.trace * (1 - self.decay)
			self.measurement.append(avg_input)

	def plot(self, title: str = None):
		"""Plot the stored EMA traces with ``tt.plot.line_graph``."""
		plot_title = title if title is not None else self.title
		rounded_list = [round(x, 4) for x in self.decay.tolist()]
		plot.line_graph(self.measurement, title=plot_title, label=rounded_list)
