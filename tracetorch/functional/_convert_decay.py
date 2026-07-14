import math
import torch
from typing import Literal, Union

metric_type = Literal["decay, halflife, tau, horizon"]


def convert_decay(
        value: Union[float, torch.Tensor],
        from_type: metric_type,
        to_type: metric_type,
):
    is_tensor = torch.is_tensor(value)
    m = torch if is_tensor else math

    if from_type == "decay":
        decay = value
    elif from_type == "halflife":
        decay = m.exp(-m.log(2) / value)
    elif from_type == "tau":
        decay = m.exp(-1.0 / value)
    elif from_type == "horizon":
        decay = 1.0 - (1.0 / value)
    else:
        raise ValueError(f"Unknown source metric: {from_type}.")

    if is_tensor:
        is_valid = ((decay > 0.0) & (decay < 1.0)).all()
    else:
        is_valid = 0.0 < decay < 1.0

    if not is_valid:
        raise ValueError(f"Calculated base decay outside (0,1) range: {decay}. Check input value is valid.")

    if to_type == "decay":
        return decay
    elif to_type == "halflife":
        return -m.log(2) / m.log(decay)
    elif to_type == "tau":
        return -1.0 / m.log(decay)
    elif to_type == "horizon":
        return 1.0 / (1.0 - decay)
    else:
        raise ValueError(f"Unknown target metric: {to_type}.")
