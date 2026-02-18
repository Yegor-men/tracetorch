from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from .._tt_infrastructure import TTLayer
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
    passthrough: bool
    learnable: bool


class VectorConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    learnable: bool


DEFAULT_ALPHA = {"value": 0.5, "rank": 1, "ema": True, "learnable": True}
DEFAULT_BETA = {"value": 0.9, "rank": 1, "ema": False, "learnable": True}
DEFAULT_GAMMA = {"value": 0.9, "rank": 1, "ema": True, "learnable": True}
DEFAULT_THRESHOLD = {"value": 1.0, "rank": 1, "surrogate": atan_surrogate(2.0), "passthrough": False, "learnable": True}
DEFAULT_SCALE = {"value": 1.0, "rank": 1, "learnable": True}
DEFAULT_REC_WEIGHT = {"value": 0.0, "rank": 1, "learnable": True}
DEFAULT_BIAS = {"value": 0.0, "rank": 1, "learnable": True}


class LeakyIntegrator(TTLayer):
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
            pos_rec_weight_setup: Optional[VectorConfig] = None,
            neg_rec_weight_setup: Optional[VectorConfig] = None,
            bias_setup: Optional[VectorConfig] = None,

    ):
        super().__init__(num_neurons, dim)

        def _setup_decay(
                name: str,
                pos_setup: Optional[DecayConfig],
                neg_setup: Optional[DecayConfig],
                default_config: dict
        ):
            setattr(self, f"use_{name}", pos_setup is not None or neg_setup is not None)

            if getattr(self, f"use_{name}"):
                setattr(self, f"dual_{name}", pos_setup is not None and neg_setup is not None)
                state_mapping = {"alpha": "syn", "beta": "mem", "gamma": "rec"}
                state_name = state_mapping[name]
                if getattr(self, f"dual_{name}"):  # if duality
                    self._initialize_state(f"pos_{state_name}")
                    self._initialize_state(f"neg_{state_name}")
                    pos_cfg = {**default_config, **(pos_setup or {})}
                    neg_cfg = {**default_config, **(neg_setup or {})}
                    self._register_decay(f"pos_{name}", pos_cfg["value"], pos_cfg["rank"], pos_cfg["learnable"])
                    self._register_decay(f"neg_{name}", neg_cfg["value"], neg_cfg["rank"], neg_cfg["learnable"])
                    setattr(self, f"ema_pos_{state_name}", pos_cfg["ema"])
                    setattr(self, f"ema_neg_{state_name}", pos_cfg["ema"])
                else:
                    self._initialize_state(f"{state_name}")
                    setup = pos_setup or neg_setup
                    cfg = {**default_config, **(setup or {})}
                    self._register_decay(f"{name}", cfg["value"], cfg["rank"], cfg["learnable"])
                    setattr(self, f"ema_{state_name}", cfg["ema"])

        _setup_decay("alpha", pos_alpha_setup, neg_alpha_setup, DEFAULT_ALPHA)
        _setup_decay("beta", pos_beta_setup, neg_beta_setup, DEFAULT_BETA)
        _setup_decay("gamma", pos_gamma_setup, neg_gamma_setup, DEFAULT_GAMMA)

        assert self.use_beta, "beta must be initialized"

        if self.use_gamma:
            self._initialize_state("prev_output")

        def _setup_threshold(
                name: str,
                setup: Optional[DecayConfig],
                default_config: dict
        ):
            cfg = {**default_config, **(setup or {})}
            setattr(self, f"use_{name}", setup is not None)
            if getattr(self, f"use_{name}"):
                setattr(self, f"passthrough_{name}", cfg["passthrough"])
                if not getattr(self, f"passthrough_{name}"):  # if not passthrough
                    self._register_threshold(f"{name}", cfg["value"], cfg["rank"], cfg["learnable"])
                    setattr(self, f"{name}_heaviside", cfg["surrogate"])

        _setup_threshold("pos_threshold", pos_threshold_setup, DEFAULT_THRESHOLD)
        _setup_threshold("neg_threshold", neg_threshold_setup, DEFAULT_THRESHOLD)

        def _setup_vector(
                name: str,
                pos_setup: Optional[DecayConfig],
                neg_setup: Optional[DecayConfig],
                default_config: dict
        ):
            setattr(self, f"use_{name}", pos_setup is not None or neg_setup is not None)
            if getattr(self, f"use_{name}"):
                setattr(self, f"dual_{name}", pos_setup is not None and neg_setup is not None)
                if getattr(self, f"dual_{name}"):
                    pos_cfg = {**default_config, **(pos_setup or {})}
                    neg_cfg = {**default_config, **(neg_setup or {})}
                    self._register_parameter(f"pos_{name}", pos_cfg["value"], pos_cfg["rank"], pos_cfg["learnable"])
                    self._register_parameter(f"neg_{name}", neg_cfg["value"], neg_cfg["rank"], neg_cfg["learnable"])
                else:
                    setup = pos_setup or neg_setup
                    cfg = {**default_config, **(setup or {})}
                    self._register_parameter(f"{name}", cfg["value"], cfg["rank"], cfg["learnable"])

        _setup_vector("scale", pos_scale_setup, neg_scale_setup, DEFAULT_SCALE)
        _setup_vector("rec_weight", pos_rec_weight_setup, neg_rec_weight_setup, DEFAULT_REC_WEIGHT)
        _setup_vector("bias", bias_setup, None, DEFAULT_BIAS)

        if self.use_rec_weight:
            assert self.use_gamma, "rec_weight is initialized, but gamma decay and rec trace is not"

    def forward(self, x):
        self._ensure_states(x)
        x_moved = self._to_working_dim(x)
        mem_delta = torch.zeros_like(x)

        if self.use_alpha:
            if self.dual_alpha:
                pos_syn_moved = self._to_working_dim(self.pos_syn)
                neg_syn_moved = self._to_working_dim(self.neg_syn)

                pos_syn_moved_delta = torch.where(x_moved >= 0, x_moved, 0.0)
                neg_syn_moved_delta = torch.where(x_moved <= 0, x_moved, 0.0)

                if self.ema_pos_syn:
                    pos_syn_moved_delta = pos_syn_moved_delta * (1 - self.pos_alpha)

                if self.ema_neg_syn:
                    neg_syn_moved_delta = neg_syn_moved_delta * (1 - self.neg_alpha)

                pos_syn_moved = pos_syn_moved * self.pos_alpha + pos_syn_moved_delta
                neg_syn_moved = neg_syn_moved * self.neg_alpha + neg_syn_moved_delta

                mem_delta = mem_delta + pos_syn_moved + neg_syn_moved

                self.pos_syn = self._from_working_dim(pos_syn_moved)
                self.neg_syn = self._from_working_dim(neg_syn_moved)
            else:
                syn_moved = self._to_working_dim(self.syn)

                syn_moved_delta = x_moved

                if self.ema_syn:
                    syn_moved_delta = syn_moved_delta * (1 - self.alpha)

                syn_moved = syn_moved * self.alpha + syn_moved_delta

                mem_delta = mem_delta + syn_moved

                self.syn = self._from_working_dim(syn_moved)
        else:
            mem_delta = x_moved

        if self.use_gamma:
            prev_output_moved = self._to_working_dim(self.prev_output)
            if self.dual_gamma:
                pos_rec_moved = self._to_working_dim(self.pos_rec)
                neg_rec_moved = self._to_working_dim(self.neg_rec)

                pos_rec_moved_delta = torch.where(prev_output_moved >= 0, prev_output_moved, 0.0)
                neg_rec_moved_delta = torch.where(prev_output_moved <= 0, prev_output_moved, 0.0)

                # rec_weight scaling must happen BEFORE integration into mem_delta, for mathematical reasons
                if self.use_rec_weight:
                    if self.dual_rec_weight:
                        pos_rec_moved_delta = pos_rec_moved_delta * self.pos_rec_weight
                        neg_rec_moved_delta = neg_rec_moved_delta * self.neg_rec_weight
                    else:
                        pos_rec_moved_delta = pos_rec_moved_delta * self.rec_weight
                        neg_rec_moved_delta = neg_rec_moved_delta * self.rec_weight

                if self.ema_pos_rec:
                    pos_rec_moved_delta = pos_rec_moved_delta * (1 - self.pos_gamma)

                if self.ema_neg_rec:
                    neg_rec_moved_delta = neg_rec_moved_delta * (1 - self.neg_gamma)

                pos_rec_moved = pos_rec_moved * self.pos_gamma + pos_rec_moved_delta
                neg_rec_moved = neg_rec_moved * self.neg_gamma + neg_rec_moved_delta

                mem_delta = mem_delta + pos_rec_moved + neg_rec_moved

                self.pos_rec = self._from_working_dim(pos_rec_moved)
                self.neg_rec = self._from_working_dim(neg_rec_moved)
            else:
                rec_moved = self._to_working_dim(self.rec)

                pos_rec_moved_delta = torch.where(prev_output_moved >= 0, prev_output_moved, 0.0)
                neg_rec_moved_delta = torch.where(prev_output_moved <= 0, prev_output_moved, 0.0)

                # rec_weight scaling must happen BEFORE integration into mem_delta, for mathematical reasons
                if self.use_rec_weight:
                    if self.dual_rec_weight:
                        pos_rec_moved_delta = pos_rec_moved_delta * self.pos_rec_weight
                        neg_rec_moved_delta = neg_rec_moved_delta * self.neg_rec_weight
                    else:
                        pos_rec_moved_delta = pos_rec_moved_delta * self.rec_weight
                        neg_rec_moved_delta = neg_rec_moved_delta * self.rec_weight

                rec_moved_delta = pos_rec_moved_delta + neg_rec_moved_delta

                if self.ema_rec:
                    rec_moved_delta = rec_moved_delta * (1 - self.gamma)

                rec_moved = rec_moved * self.gamma + rec_moved_delta

                mem_delta = mem_delta + rec_moved

                self.rec = self._from_working_dim(rec_moved)

        if self.use_bias:
            mem_delta = mem_delta + self.bias

        # ASSUMED THAT BETA IS USED SINCE ASSERTED AT INIT
        if self.dual_beta:
            pos_mem_moved = self._to_working_dim(self.pos_mem)
            neg_mem_moved = self._to_working_dim(self.neg_mem)

            pos_mem_moved_delta = torch.where(mem_delta >= 0, mem_delta, 0.0)
            neg_mem_moved_delta = torch.where(mem_delta <= 0, mem_delta, 0.0)

            if self.ema_pos_mem:
                pos_mem_moved_delta = pos_mem_moved_delta * (1 - self.pos_beta)

            if self.ema_neg_mem:
                neg_mem_moved_delta = neg_mem_moved_delta * (1 - self.neg_beta)

            pos_mem_moved = pos_mem_moved * self.pos_beta + pos_mem_moved_delta
            neg_mem_moved = neg_mem_moved * self.neg_beta + neg_mem_moved_delta

            mem_moved = pos_mem_moved + neg_mem_moved
        else:
            mem_moved = self._to_working_dim(self.mem)

            if self.ema_mem:
                mem_delta = mem_delta * (1 - self.beta)

            mem_moved = mem_moved * self.beta + mem_delta

        pos_output_moved = torch.zeros_like(mem_moved)
        neg_output_moved = torch.zeros_like(mem_moved)

        if self.use_pos_threshold or self.use_neg_threshold:
            if self.use_pos_threshold:
                if self.passthrough_pos_threshold:
                    pos_output_moved = torch.where(mem_moved >= 0, mem_moved, 0.0)
                else:
                    pos_output_moved = self.pos_threshold_heaviside(mem_moved - self.pos_threshold)

            if self.use_neg_threshold:
                if self.passthrough_neg_threshold:
                    neg_output_moved = torch.where(mem_moved <= 0, mem_moved, 0.0)
                else:
                    neg_output_moved = -self.neg_threshold_heaviside(self.neg_threshold - mem_moved)

            if self.use_pos_threshold and not self.passthrough_pos_threshold:
                threshold_subtraction = pos_output_moved * self.pos_threshold
                if self.dual_beta:
                    pos_mem_moved = pos_mem_moved - threshold_subtraction * 0.5
                    neg_mem_moved = neg_mem_moved - threshold_subtraction * 0.5
                else:
                    mem_moved = mem_moved - threshold_subtraction

            if self.use_neg_threshold and not self.passthrough_neg_threshold:
                threshold_subtraction = neg_output_moved * self.neg_threshold
                if self.dual_beta:
                    pos_mem_moved = pos_mem_moved - threshold_subtraction * 0.5
                    neg_mem_moved = neg_mem_moved - threshold_subtraction * 0.5
                else:
                    mem_moved = mem_moved - threshold_subtraction

        else:
            pos_output_moved = torch.where(mem_moved >= 0, mem_moved, 0.0)
            neg_output_moved = torch.where(mem_moved <= 0, mem_moved, 0.0)

        # to decouple scale from rec_weight, these things happen separately
        if self.use_gamma:
            self.prev_output = self._from_working_dim(pos_output_moved + neg_output_moved)

        if self.use_scale:
            if self.dual_scale:
                pos_output_moved = pos_output_moved * self.pos_scale
                neg_output_moved = neg_output_moved * self.neg_scale
            else:
                pos_output_moved = pos_output_moved * self.scale
                neg_output_moved = neg_output_moved * self.scale

        output_moved = pos_output_moved + neg_output_moved
        output = self._from_working_dim(output_moved)

        if self.dual_beta:
            self.pos_mem = self._from_working_dim(pos_mem_moved)
            self.neg_mem = self._from_working_dim(neg_mem_moved)
        else:
            self.mem = self._from_working_dim(mem_moved)

        return output
