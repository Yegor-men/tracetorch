import torch
import matplotlib.pyplot as plt
import numpy as np


def spike_train(
		list_of_tensors,
		spacing: float = 1.0,
		linelength: float = 0.8,
		linewidth: float = 0.5,
		title: str = "Spike Train Raster",
		use_imshow: bool = True,
):
	"""
	Plot a spike train raster for a list of 1D tensors.

	If use_imshow is True, uses imshow with grayscale to reflect values
	(0=white, 1=black). Otherwise falls back to eventplot, plotting spikes
	(only non-zero) in black.

	Args:
		list_of_tensors: list of 1D torch.Tensors of equal length T.
		spacing: vertical spacing between rows.
		linelength: length of eventplot lines (ignored for imshow).
		linewidth: width of eventplot lines (ignored for imshow).
		title: figure title.
		use_imshow: whether to plot using imshow and grayscale mapping.
	:param list_of_tensors:
	:param spacing:
	:param linelength:
	:param linewidth:
	:param title:
	:param use_imshow:
	:return:
	"""
	# Stack into 2D array: shape (num_neurons, T)
	# list_of_tensors assumed length T each
	data = torch.stack(list_of_tensors).cpu().detach().numpy()  # shape (T, N)
	data = data.T  # shape (N, T)

	plt.figure(figsize=(8, 4))
	if use_imshow:
		# Display as image with grayscale: 1=black, 0=white
		plt.imshow(
			data,
			aspect='auto',
			cmap='gray_r',
			origin='lower',
			interpolation='nearest'
		)
		plt.colorbar(label='Spike value')
	else:
		N, T = data.shape
		# for each neuron i, list of spike times where value != 0
		spike_times = [
			[t for t in range(T) if data[i, t] != 0]
			for i in range(N)
		]
		# y offsets
		offsets = np.arange(N) * spacing
		plt.eventplot(
			spike_times,
			orientation='horizontal',
			lineoffsets=offsets,
			linelengths=linelength,
			linewidths=linewidth,
			colors='k'
		)
		plt.ylim(-spacing, offsets[-1] + spacing)

	plt.xlabel("Time step")
	plt.ylabel("Neuron index")
	plt.title(title)
	plt.tight_layout()
	plt.show()
