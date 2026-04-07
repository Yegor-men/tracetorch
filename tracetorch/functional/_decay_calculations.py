import math
import torch


def halflife_to_decay(num_steps: float):
    return math.exp(-math.log(2) / num_steps)


def decay_to_halflife(decay: float):
    return -math.log(2) / math.log(decay)


def timesteps_to_decay(time: float):
    return 1 - 1 / time


def decay_to_timesteps(decay: float):
    return 1 / (1 - decay)


def hippo_decays(N: int, scale: float = 0.01, device=None, dtype=torch.float32, shuffle: bool = False):
    """
    Returns a vector of N values with exactly the HiPPO diagonal spectrum

    Slowest decay = exp(-1 * scale) -> long memory
    Fastest decay = exp(-N * scale) -> short memory
    """
    rates = torch.arange(1, N + 1, dtype=dtype, device=device)
    decays = torch.exp(-rates * scale)
    if shuffle:
        idx = torch.randperm(N, device=device)
        decays = decays[idx]
    return decays


def max_halflife_to_hippo_scale(half_life: float):
    return math.log(2) / half_life
