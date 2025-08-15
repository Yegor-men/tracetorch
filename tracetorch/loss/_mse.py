import torch


def mse(
		received: torch.Tensor,
		expected: torch.Tensor,
		reduction: str = "mean"
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	calculates the mse loss
	:param received:
	:param expected:
	:param reduction:
	:return:
	"""
	received.requires_grad_(True)

	loss_fn = torch.nn.MSELoss(reduction=reduction)
	loss = loss_fn.forward(received, expected)
	loss.backward()

	ls = received.grad.detach().clone()

	return loss, ls
