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
        beta_setup = {
            "value": beta,
            "rank": beta_rank,
            "ema": True,
            "learnable": learn_beta,
        }

        threshold_setup = {
            "value": threshold,
            "rank": threshold_rank,
            "surrogate": surrogate_derivative,
            "learnable": learn_threshold,
        }

        super().__init__(
            num_neurons=num_neurons,
            dim=dim,
            pos_beta_setup=beta_setup,
            pos_threshold_setup=threshold_setup,
        )


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
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_threshold: bool = True,
            learn_rec_weight: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class DSLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_threshold: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class DRLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_threshold: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class SRLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_threshold: bool = True,
            learn_rec_weight: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass


class DSRLIB(LeakyIntegrator):
    def __init__(
            self,
            num_neurons: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_threshold: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            surrogate_derivative=functional.atan_surrogate(2.0),
    ):
        pass
