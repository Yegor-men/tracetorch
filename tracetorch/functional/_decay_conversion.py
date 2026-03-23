import math


def halflife_to_decay(num_steps: float):
    return math.exp(-math.log(2) / num_steps)


def decay_to_halflife(decay: float):
    return -math.log(2) / math.log(decay)


def timesteps_to_decay(time: float):
    return 1 - 1 / time


def decay_to_timesteps(decay: float):
    return 1 / (1 - decay)
