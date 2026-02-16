from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from .._tt_infrastructure import TTLayer, TTModel
from ...functional import atan_surrogate, sigmoid_surrogate


class DecayConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    ema: bool
    learnable: bool


class ThresholdConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    surrogate: Any  # has to be a surrogate function
    learnable: bool


class VectorConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    learnable: bool


DEFAULT_ALPHA = {"value": 0.5, "rank": 1, "ema": True, "learnable": True}
DEFAULT_BETA = {"value": 0.9, "rank": 1, "ema": False, "learnable": True}
DEFAULT_GAMMA = {"value": 0.9, "rank": 1, "ema": True, "learnable": True}
DEFAULT_THRESHOLD = {"value": 1.0, "rank": 1, "surrogate": atan_surrogate(2.0), "learnable": True}
DEFAULT_SCALE = {"value": 1.0, "rank": 1, "learnable": True}
DEFAULT_REC_WEIGHT = {"value": 0.0, "rank": 1, "learnable": True}
DEFAULT_BIAS = {"value": 0.0, "rank": 1, "learnable": True}


class LeakyIntegrator(TTLayer, TTModel):
    def __init__(
            self,
            num_neurons: int,
            dim: int = -1,

            pos_alpha_setup: Optional[DecayConfig] = None,
            neg_alpha_setup: Optional[DecayConfig] = None,
            pos_beta_setup: Optional[DecayConfig] = DEFAULT_BETA,
            neg_beta_setup: Optional[DecayConfig] = None,
            pos_gamma_setup: Optional[DecayConfig] = None,
            neg_gamma_setup: Optional[DecayConfig] = None,
            pos_threshold_setup: Optional[ThresholdConfig] = None,
            neg_threshold_setup: Optional[ThresholdConfig] = None,
            pos_scale_setup: Optional[VectorConfig] = None,
            neg_scale_setup: Optional[VectorConfig] = None,
            rec_weight_setup: Optional[VectorConfig] = None,
            bias_setup: Optional[VectorConfig] = None,

    ):
        TTLayer.__init__(self, num_neurons, dim)
        TTModel.__init__(self)
