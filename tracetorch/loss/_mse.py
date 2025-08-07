import torch


def mse(
		received: torch.Tensor,
		expected: torch.Tensor,
		reduction: str = "sum"
) -> tuple[torch.Tensor, torch.Tensor]:
	received = received.detach().clone().requires_grad_(True)
	expected = expected.detach().to(received)

	loss_fn = torch.nn.MSELoss(reduction=reduction)
	loss = loss_fn.forward(received, expected)
	loss.backward()

	ls = received.grad.detach()

	return loss.detach(), ls
