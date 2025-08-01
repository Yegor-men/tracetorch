import torch
import matplotlib.pyplot as plt


def line_graph(list_of_tensors: list[torch.Tensor], title: str) -> None:
	T = len(list_of_tensors)
	N = list_of_tensors[0].numel()

	# stack into shape (T, …)
	data = torch.stack(list_of_tensors, dim=0)  # if scalars, shape == (T,)
	# quick fix: if it came out 1-D, treat as (T,1)
	if data.dim() == 1:
		data = data.unsqueeze(1)  # now shape == (T,1)

	data = data.cpu().numpy()
	x = range(T)

	for neuron_idx in range(N):
		plt.plot(x, data[:, neuron_idx], label=f'Neuron {neuron_idx}')

	plt.title(title)
	if N > 1:
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.show()
