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
            ema_beta: bool = False,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": ema_beta,
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
            ema_beta: bool = False,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
    ):
        pos_beta_setup = {
            "value": pos_beta,
            "rank": pos_beta_rank,
            "ema": ema_beta,
            "learnable": learn_pos_beta,
        }

        neg_beta_setup = {
            "value": neg_beta,
            "rank": neg_beta_rank,
            "ema": ema_beta,
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
            ema_beta: bool = False,
            learn_alpha: bool = True,
            learn_beta: bool = True,
    ):
        alpha_setup = {
            "value": alpha,
            "rank": alpha_rank,
            "ema": True,
            "learnable": learn_alpha,
        }

        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": ema_beta,
            "learnable": learn_beta,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_alpha_setup=alpha_setup,
            pos_beta_setup=beta_setup,
        )
