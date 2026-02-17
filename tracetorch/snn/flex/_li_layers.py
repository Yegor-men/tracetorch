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
        pos_beta_setup = {
            "value": pos_beta,
            "rank": pos_beta_rank,
            "ema": True,
            "learnable": learn_pos_beta,
        }

        neg_beta_setup = {
            "value": neg_beta,
            "rank": neg_beta_rank,
            "ema": True,
            "learnable": learn_neg_beta,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_beta_setup=pos_beta_setup,
            neg_beta_setup=neg_beta_setup,
        )


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
        alpha_setup = {
            "value": alpha,
            "rank": alpha_rank,
            "ema": False,
            "learnable": learn_alpha,
        }

        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": True,
            "learnable": learn_beta,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_alpha_setup=alpha_setup,
            pos_beta_setup=beta_setup,
        )


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
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": True,
            "learnable": learn_beta,
        }

        gamma_setup = {
            "value": gamma,
            "rank": gamma_rank,
            "ema": True,
            "learnable": learn_gamma,
        }

        rec_weight_setup = {
            "value": rec_weight,
            "rank": rec_weight_rank,
            "learnable": learn_rec_weight,
        }

        bias_setup = {
            "value": bias,
            "rank": bias_rank,
            "learnable": learn_bias,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_beta_setup=beta_setup,
            gamma_setup=gamma_setup,
            rec_weight_setup=rec_weight_setup,
            bias_setup=bias_setup,
        )
