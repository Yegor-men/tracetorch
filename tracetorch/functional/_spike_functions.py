import torch


def sigmoid4x(x: torch.Tensor) -> torch.Tensor:
    r"""Apply the default traceTorch smooth spike function.

    ``sigmoid4x`` is a steeper sigmoid used to turn membrane distance from
    threshold into a differentiable firing intensity. SNN firing layers pass
    values such as ``mem - threshold + bias`` through this function to decide
    how much charge to emit and reset.

    Args:
        x (torch.Tensor): membrane distance from threshold.

    Returns:
        torch.Tensor: smooth firing intensity in ``(0, 1)``.
    """
    return torch.sigmoid(4.0 * x)


def _sigmoid4x_backward(x: torch.Tensor) -> torch.Tensor:
    y = sigmoid4x(x)
    return 4.0 * y * (1.0 - y)


class RoundSigmoid4x(torch.autograd.Function):
    r"""Hard deterministic spike with a ``sigmoid4x`` surrogate derivative."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * _sigmoid4x_backward(x)


def round_sigmoid4x(x: torch.Tensor) -> torch.Tensor:
    r"""Apply a deterministic hard spike with a ``sigmoid4x`` surrogate.

    The forward pass returns ``1`` when the membrane distance from threshold is
    non-negative and ``0`` otherwise. The backward pass uses the derivative of
    ``sigmoid4x`` so the maximum surrogate gradient is ``1`` at threshold.
    """
    return RoundSigmoid4x.apply(x)


class StochasticSigmoid4x(torch.autograd.Function):
    r"""Bernoulli hard spike with a ``sigmoid4x`` surrogate derivative."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.bernoulli(sigmoid4x(x))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * _sigmoid4x_backward(x)


def stochastic_sigmoid4x(x: torch.Tensor) -> torch.Tensor:
    r"""Apply a stochastic hard spike with a ``sigmoid4x`` surrogate.

    The forward pass samples from ``Bernoulli(sigmoid4x(x))``. The backward
    pass uses the derivative of ``sigmoid4x`` so the maximum surrogate gradient
    is ``1`` at threshold.
    """
    return StochasticSigmoid4x.apply(x)
