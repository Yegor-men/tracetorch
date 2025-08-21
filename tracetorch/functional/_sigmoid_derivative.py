import torch
from torch.nn.functional import sigmoid


def sigmoid_derivative(tensor: torch.Tensor):
	return sigmoid(tensor) * (1 - sigmoid(tensor))
