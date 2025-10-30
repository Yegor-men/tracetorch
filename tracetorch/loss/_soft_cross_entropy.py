import torch


def soft_cross_entropy(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8, reduction: str = 'mean'):
	"""
	Generalized soft cross entropy between two probability tensors of the same shape.

	pred:   [B, ...] predicted probabilities (e.g., from softmax)
	target: [B, ...] target probabilities (e.g., one-hot, Dirichlet samples, etc.)
	eps:    Small value for numerical stability
	reduction: 'none' | 'mean' | 'sum'
	"""
	if pred.shape != target.shape:
		raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

	pred = torch.clamp(pred, eps, 1.0)
	loss = -(target * torch.log(pred))

	# Sum over all non-batch dims
	loss = loss.view(loss.size(0), -1).sum(dim=1)

	if reduction == 'mean':
		return loss.mean()
	elif reduction == 'sum':
		return loss.sum()
	elif reduction == 'none':
		return loss
	else:
		raise ValueError(f"Invalid reduction: {reduction}")
