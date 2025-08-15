import torch

def sigmoid_inverse(tensor: torch.Tensor) -> torch.Tensor:
	"""
	Calculates the necessary tensor such that passing it through sigmoid yields this tensor
	:param tensor:
	:return:
	"""
	with torch.no_grad():
		return torch.log(tensor / (1 - tensor))