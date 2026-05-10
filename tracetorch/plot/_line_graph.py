import torch
import matplotlib.pyplot as plt


def line_graph(list_of_values, title: str, label=None) -> None:
	"""Plot a simple line graph from scalar values or tensors.

	Args:
		list_of_values: sequence of Python scalars or tensors. Tensor values are
			stacked over time and each flattened element is plotted as its own
			line.
		title (str): plot title.
		label (Sequence, optional): labels for tensor-valued lines.

	Notes:
		This helper is intended for quick experiment visualization. It calls
		``plt.show()`` and does not return the figure.
	"""
	# Check if it's a list of tensors
	if isinstance(list_of_values[0], torch.Tensor):
		T = len(list_of_values)
		N = list_of_values[0].numel()

		# stack into shape (T, …)
		data = torch.stack(list_of_values, dim=0)  # if scalars, shape == (T,)
		if data.dim() == 1:
			data = data.unsqueeze(1)  # now shape == (T,1)

		data = data.cpu().numpy()
		x = range(T)

		for neuron_idx in range(N):
			if label is not None:
				plt.plot(x, data[:, neuron_idx], label=f'{label[neuron_idx]}')
			else:
				plt.plot(x, data[:, neuron_idx], label=f'Neuron {neuron_idx}')

		plt.title(title)
		if N > 1:
			plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	else:
		# Assume list of floats (or ints)
		x = range(len(list_of_values))
		plt.plot(x, list_of_values)
		plt.title(title)

	plt.tight_layout()
	plt.show()
