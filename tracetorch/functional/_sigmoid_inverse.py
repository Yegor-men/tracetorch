import torch

def sigmoid_inverse(tensor: torch.Tensor) -> torch.Tensor:
	return torch.log(tensor / (1 - tensor))