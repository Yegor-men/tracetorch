from typing import TypedDict, Optional, Literal, Union, Dict, Any
from ._leaky_integrator import LeakyIntegrator
from ... import functional
import torch
from torch import nn


class LIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class DLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class SLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class RLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            learn_gamma: bool = True,
            learn_rec_weight: bool = True,
            learn_bias: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass
