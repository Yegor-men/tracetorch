import torch


def cross_entropy(
		received: torch.Tensor,
		expected: torch.Tensor
):
	received.requires_grad_(True).retain_grad()
	expected.requires_grad_(True)
	# loss = torch.sum()
	loss = torch.nn.functional.cross_entropy(received, expected)
	loss.backward()
	learning_signal = received.grad.detach().clone()
	learning_signal.grad = None
	del received
	return loss, learning_signal
