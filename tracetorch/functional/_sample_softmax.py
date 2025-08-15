import torch


def sample_softmax(probability_dist, return_one_hot: bool = True):
	with torch.no_grad():
		index = torch.multinomial(probability_dist, num_samples=1)
		if not return_one_hot:
			return index
		out_spikes = torch.zeros_like(probability_dist)
		out_spikes[index] = 1
		return out_spikes
