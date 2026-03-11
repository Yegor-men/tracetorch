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
