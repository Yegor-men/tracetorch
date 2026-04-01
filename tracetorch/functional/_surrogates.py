import torch


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def round_ste():
    def inner(x: torch.Tensor):
        return RoundSTE.apply(x)

    return inner


class BernoulliSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def bernoulli_ste():
    def inner(x: torch.Tensor):
        return BernoulliSTE.apply(x)

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
