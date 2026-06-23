import torch
def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
    r"""Return the logit transform of a tensor in ``(0, 1)``.

    traceTorch uses this when registering constrained decay parameters. The raw
    stored value can be optimized freely, while ``torch.sigmoid(raw)`` recovers
    the user-facing decay.

    Args:
        x (torch.Tensor): tensor whose values must lie strictly between zero and
            one.

    Returns:
        torch.Tensor: unconstrained tensor such that ``sigmoid(output) == x`` up
        to numerical precision.
    """
    assert torch.all((x > 0) & (x < 1)), "Tensor for inverse sigmoid has values outside the (0,1) range"
    return torch.logit(x)


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    r"""Return the inverse softplus transform of a positive tensor.

    traceTorch uses this when registering positive constrained parameters such as
    SNN thresholds. The raw stored value can be optimized freely, while
    ``softplus(raw)`` recovers the positive user-facing value.

    Args:
        x (torch.Tensor): tensor whose values must be strictly positive.

    Returns:
        torch.Tensor: unconstrained tensor such that ``softplus(output) == x`` up
        to numerical precision.
    """
    assert torch.all(x > 0), "Tensor for softplus inverse has values outside the (0,inf) range"
    return torch.log(torch.expm1(x))
