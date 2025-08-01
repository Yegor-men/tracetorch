import torch

def softplus_inverse(tensor: torch.Tensor) -> torch.Tensor:
	return torch.log(torch.e ** tensor - 1)