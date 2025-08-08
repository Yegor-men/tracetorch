import torch
import matplotlib.pyplot as plt
import numpy as np


def spike_train(list_of_tensors,
				spacing: float = 1.0,
				linelength: float = 0.8,
				linewidth: float = 0.5,
				title: str = "Spike Train Raster"
				) -> None:
	N = list_of_tensors[0].numel()

	# for each neuron i, collect all t where vec[i] != 0
	spike_times = [
		[t for t, vec in enumerate(list_of_tensors) if vec[i].item() != 0]
		for i in range(N)
	]

	# y‑positions for each row
	offsets = np.arange(N) * spacing

	plt.eventplot(spike_times,
				  orientation='horizontal',
				  lineoffsets=offsets,
				  linelengths=linelength,
				  linewidths=linewidth,
				  colors='k')

	plt.xlabel("Time step")
	plt.ylabel("Neuron index")
	plt.title(title)
	plt.show()
