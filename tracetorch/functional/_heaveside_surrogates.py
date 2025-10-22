import torch


# ======================================================================================================================

class ATanSurrogate(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input_, alpha):
		ctx.save_for_backward(input_)
		ctx.alpha = float(alpha)

		out = (input_ > 0).float()
		return out

	@staticmethod
	def backward(ctx, grad_output):
		(input_,) = ctx.saved_tensors
		alpha = ctx.alpha
		# grad = (alpha / 2) / (1 + (pi/2 * alpha * u)^2) * grad_output
		denom = 1.0 + (torch.pi / 2.0 * alpha * input_).pow(2)
		grad = grad_output * (alpha / 2.0) / denom
		return grad, None


def atan_surrogate(alpha: float = 2.0):
	def inner(x: torch.Tensor):
		return ATanSurrogate.apply(x, alpha)

	return inner


# ======================================================================================================================


class SigmoidSurrogate(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input_, slope):
		ctx.save_for_backward(input_)
		ctx.slope = float(slope)

		out = (input_ > 0).float()
		return out

	@staticmethod
	def backward(ctx, grad_output):
		(input_,) = ctx.saved_tensors
		k = ctx.slope

		s = torch.sigmoid(k * input_)
		grad_input = grad_output * (k * s * (1.0 - s))

		return grad_input, None


def sigmoid_surrogate(slope: float = 25):
	def inner(x: torch.Tensor):
		return SigmoidSurrogate.apply(x, slope)

	return inner

# ======================================================================================================================
