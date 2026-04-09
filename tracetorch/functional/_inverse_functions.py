import torch
import math


def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
    """Returns a tensor such that passing it through sigmoid function (1 / (1 + e**-x)) results in the inputted tensor"""
    assert torch.all((x > 0) & (x < 1)), "Tensor for inverse sigmoid has values outside the (0,1) range"
    return torch.logit(x)


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """Returns a tensor such that passing it through softplus function (ln(1 + e**x)) results in the inputted tensor"""
    assert torch.all(x > 0), "Tensor for softplus inverse has values outside the (0,inf) range"
    return torch.log(torch.expm1(x))


def mamba_scale(x: torch.Tensor) -> torch.Tensor:
    """Returns a tensor such that passing it through the mamba scale function (exp(ln(2) * -exp(scale))) results in the inputted tensor"""
    assert torch.all((x > 0) & (x < 1)), "Tensor for mamba scale has values outside the (0,1) range"
    return torch.log(-torch.log(x) / math.log(2.0))
