import torch


def logprob(distribution: torch.Tensor, sample: torch.Tensor):
	distribution.add_(1e-6)
	distribution.requires_grad_(True)
	distribution.retain_grad()
	loss = (-torch.log(distribution) * sample).sum()
	loss.backward()
	ls = distribution.grad.detach().clone()
	distribution.grad = None
	del distribution
	return loss, ls
