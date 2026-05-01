import math
import torch


def halflife_to_decay(num_steps: float):
    r"""Convert a halflife (how many timesteps are needed to lose half the signal) into a decay (per-timestep multiplicative value).

    Mathematically, solves :math:`d^{\text{num steps}}=0.5` for :math:`d`.

    Args:
        num_steps (float): the number of steps needed for the signal to halve in magnitude.

    Examples::

        >>> print(halflife_to_decay(1))
        0.5

        >>> print(halflife_to_decay(10))
        0.933032991537

        >>> print(halflife_to_decay(100))
        0.993092495437
    """
    return math.exp(-math.log(2) / num_steps)


def decay_to_halflife(decay: float):
    r"""Convert a decay (per-timestep multiplicative value) into a halflife (how many timesteps are needed to lose half the signal).

    Mathematically, solves :math:`d^{\text{num steps}}=0.5` for :math:`\text{num steps}`.

    Args:
        decay (float): the per-timestep multiplicative value.

    Examples::

        >>> print(decay_to_halflife(0.5))
        1

        >>> print(decay_to_halflife(0.9))
        6.57881347896

        >>> print(decay_to_halflife(0.99))
        68.9675639365
    """
    return -math.log(2) / math.log(decay)


def timesteps_to_decay(time: float):
    r"""Converts a time horizon (how many timesteps a signal persists) into a decay. This is not equivalent to halflife.

    Args:
        time (float): how many timesteps a signal persists for.

    Examples::

        >>> print(timesteps_to_decay(2))
        0.5

        >>> print(timesteps_to_decay(10))
        0.9

        >>> print(timesteps_to_decay(100))
        0.99
    """
    return 1 - 1 / time


def decay_to_timesteps(decay: float):
    r"""Convert a decay into a time horizon: over how many timesteps the signal persists. This is not equivalent to halflife.

    Args:
        decay (float): the per-timestep multiplicative value.

    Examples::

        >>> print(decay_to_timesteps(0.5))
        2

        >>> print(decay_to_timesteps(0.9))
        10

        >>> print(decay_to_timesteps(0.99))
        100
    """
    return 1 / (1 - decay)
