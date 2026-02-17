from typing import TypedDict, Optional, Literal, Union, Dict, Any
from ._leaky_integrator import LeakyIntegrator
import torch
from torch import nn


class LI(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": True,
            "learnable": learn_beta,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_beta_setup=beta_setup,
        )


class DLI(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
    ):
        pass


class SLI(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
    ):
        pass


class RLI(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_rec_weight: bool = True,
            learn_bias: bool = True,
    ):
        pass
