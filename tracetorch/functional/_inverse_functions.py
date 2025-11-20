import torch


def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
	assert torch.all((x > 0) & (x < 1)), "Tensor for inverse sigmoid has values outside the (0,1) range"
	return torch.logit(x)


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
	assert torch.all(x > 0), "Tensor for softplus inverse has values outside the (0,inf) range"
	return torch.log(torch.expm1(x))
