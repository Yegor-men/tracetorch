import torch


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size):
        # 1. Scale by the inverse of the step size (e.g., 0.1 -> 10)
        scaled = x / step_size

        # 2. Perform deterministic rounding
        rounded = torch.round(scaled)

        # 3. Scale back down
        return rounded * step_size

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_ste(step_size=1.0):
    def inner(x: torch.Tensor):
        return RoundSTE.apply(x, step_size)

    return inner


class StochasticRoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size):
        # 1. Scale by the inverse of the step size (e.g., 0.1 -> 10)
        scaled = x / step_size
        floor_val = torch.floor(scaled)
        fraction = scaled - floor_val

        # 2. Probabilistic rounding logic
        mask = torch.rand_like(scaled) < fraction
        rounded = floor_val + mask.float()

        # 3. Scale back down
        return rounded * step_size

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def stochastic_round_ste(step_size=1.0):
    def inner(x: torch.Tensor):
        return StochasticRoundSTE.apply(x, step_size)

    return inner


class ProbabilisticSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x


def probabilistic_ste():
    def inner(x: torch.Tensor):
        return ProbabilisticSTE.apply(x)

    return inner
