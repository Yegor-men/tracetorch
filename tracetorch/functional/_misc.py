import torch


def uniform(tensor: torch.Tensor, a, b) -> torch.Tensor:
    r"""Map a unit-interval tensor to the interval ``[a, b]``.

    This is useful for turning ``torch.rand`` samples into a custom uniform
    range while preserving shape, device, dtype, and broadcasting behavior.

    Args:
        tensor (torch.Tensor): input tensor, typically sampled from
            ``torch.rand``.
        a: lower bound of the target interval.
        b: upper bound of the target interval.

    Returns:
        torch.Tensor: ``tensor * (b - a) + a``.
    """
    a = torch.as_tensor(a, dtype=tensor.dtype, device=tensor.device)
    b = torch.as_tensor(b, dtype=tensor.dtype, device=tensor.device)
    return tensor * (b - a) + a
