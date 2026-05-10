import torch


class RoundSTE(torch.autograd.Function):
    r"""Straight-through deterministic rounding autograd function.

    The forward pass rounds to the nearest multiple of ``step_size``. The
    backward pass returns the upstream gradient unchanged, implementing the
    straight-through estimator.
    """

    @staticmethod
    def forward(ctx, x, step_size):
        scaled = x / step_size
        rounded = torch.round(scaled)
        return rounded * step_size

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_ste(step_size=1.0):
    r"""Create a deterministic rounding straight-through quantizer.

    Args:
        step_size (float, default=1.0): quantization interval. A value of ``1``
            maps probabilities to integer-like events.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: function that rounds in the
        forward pass and passes gradients through unchanged.
    """

    def inner(x: torch.Tensor):
        return RoundSTE.apply(x, step_size)

    return inner


class StochasticRoundSTE(torch.autograd.Function):
    r"""Straight-through stochastic rounding autograd function.

    The forward pass rounds each value up with probability equal to its
    fractional part after scaling. The backward pass returns the upstream
    gradient unchanged.
    """

    @staticmethod
    def forward(ctx, x, step_size):
        scaled = x / step_size
        floor_val = torch.floor(scaled)
        fraction = scaled - floor_val
        mask = torch.rand_like(scaled) < fraction
        rounded = floor_val + mask.float()
        return rounded * step_size

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def stochastic_round_ste(step_size=1.0):
    r"""Create a stochastic rounding straight-through quantizer.

    Args:
        step_size (float, default=1.0): quantization interval.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: function that stochastically
        rounds in the forward pass and passes gradients through unchanged.
    """

    def inner(x: torch.Tensor):
        return StochasticRoundSTE.apply(x, step_size)

    return inner


class ProbabilisticSTE(torch.autograd.Function):
    r"""Probabilistic straight-through estimator for firing probabilities.

    The forward pass multiplies each probability by a Bernoulli sample from that
    probability. The backward pass scales the upstream gradient by ``2 * x``.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x


def probabilistic_ste():
    r"""Create the probabilistic straight-through quantizer.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: function that samples
        probability-weighted events in the forward pass and applies the custom
        probabilistic surrogate gradient in the backward pass.
    """

    def inner(x: torch.Tensor):
        return ProbabilisticSTE.apply(x)

    return inner
