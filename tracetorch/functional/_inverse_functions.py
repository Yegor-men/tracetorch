import torch


def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
	return torch.logit(x)


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
	return torch.log(torch.expm1(x))
