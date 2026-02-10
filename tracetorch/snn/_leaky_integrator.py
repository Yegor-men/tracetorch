from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ._ttmodule import TTModule
from .. import functional as tt_functional


class DecayConfig(TypedDict, total=False):
    value: Union[float, torch.Tensor]
    rank: Literal[0, 1]
    use_averaging: bool
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


DEFAULT_ALPHA = {"value": 0.5, "rank": 1, "use_averaging": True, "learnable": True}
DEFAULT_BETA = {"value": 0.9, "rank": 1, "use_averaging": False, "learnable": True}
DEFAULT_GAMMA = {"value": 0.5, "rank": 1, "use_averaging": True, "learnable": True}
DEFAULT_POS_THRESH = {"value": 1.0, "rank": 1, "surrogate": tt_functional.atan_surrogate(2.0), "learnable": True}
DEFAULT_NEG_THRESH = {"value": 1.0, "rank": 1, "surrogate": tt_functional.atan_surrogate(2.0), "learnable": True}
DEFAULT_POS_SCALE = {"value": 1.0, "rank": 1, "learnable": True}
DEFAULT_NEG_SCALE = {"value": 1.0, "rank": 1, "learnable": True}
DEFAULT_REC_WEIGHT = {"value": 0.0, "rank": 1, "learnable": True}
DEFAULT_BIAS = {"value": 0.0, "rank": 1, "learnable": True}


class LeakyIntegrator(TTModule):
    def __init__(
            self,
            num_neurons: int,
            dim: int = -1,

            alpha_setup: Optional[DecayConfig] = None,
            beta_setup: DecayConfig = DEFAULT_BETA,
            gamma_setup: Optional[DecayConfig] = None,
            pos_threshold_setup: Optional[ThresholdConfig] = None,
            neg_threshold_setup: Optional[ThresholdConfig] = None,
            pos_scale_setup: Optional[VectorConfig] = None,
            neg_scale_setup: Optional[VectorConfig] = None,
            rec_weight_setup: Optional[VectorConfig] = None,
            bias_setup: Optional[VectorConfig] = None,
    ):
        super().__init__()
        self.num_neurons = int(num_neurons)
        self.dim = int(dim)

        # MANDATORY: BETA
        # we assume that the user provides at least a partial config, then merge with default
        beta_cfg = {**DEFAULT_BETA, **(beta_setup or {})}
        self._setup_decay("beta", beta_cfg)
        self.mem_is_ema = beta_cfg["use_averaging"]

        # OPTIONAL
        self.use_alpha = alpha_setup is not None
        if self.use_alpha:
            alpha_cfg = {**DEFAULT_ALPHA, **(alpha_setup or {})}
            self._setup_decay("alpha", alpha_cfg)
            self.syn_is_ema = alpha_cfg["use_averaging"]

        self.use_gamma = gamma_setup is not None
        if self.use_gamma:
            gamma_cfg = {**DEFAULT_GAMMA, **(gamma_setup or {})}
            self._setup_decay("gamma", gamma_cfg)
            self.rec_is_ema = gamma_cfg["use_averaging"]

        # THRESHOLD CONFIGURATION
        self.use_pos_threshold = pos_threshold_setup is not None
        if self.use_pos_threshold:
            pos_threshold_cfg = {**DEFAULT_POS_THRESH, **(pos_threshold_setup or {})}
            self._setup_threshold("pos_threshold", pos_threshold_cfg)
            self.pos_surrogate = pos_threshold_cfg["surrogate"]

        self.use_neg_threshold = neg_threshold_setup is not None
        if self.use_neg_threshold:
            neg_threshold_cfg = {**DEFAULT_NEG_THRESH, **(neg_threshold_setup or {})}
            self._setup_threshold("neg_threshold", neg_threshold_cfg)
            self.neg_surrogate = neg_threshold_cfg["surrogate"]

        # SPIKE SCALING CONFIGURATION
        self.use_pos_scale = pos_scale_setup is not None
        if self.use_pos_scale:
            pos_scale_cfg = {**DEFAULT_POS_SCALE, **(pos_scale_setup or {})}
            self._setup_vector("pos_scale", pos_scale_cfg)

        self.use_neg_scale = neg_scale_setup is not None
        if self.use_neg_scale:
            neg_scale_cfg = {**DEFAULT_NEG_SCALE, **(neg_scale_setup or {})}
            self._setup_vector("neg_scale", neg_scale_cfg)

        # RECURRENT WEIGHT CONFIGURATION
        self.use_rec_weight = rec_weight_setup is not None
        if self.use_rec_weight:
            rec_weight_cfg = {**DEFAULT_REC_WEIGHT, **(rec_weight_setup or {})}
            self._setup_vector("rec_weight", rec_weight_cfg)
            assert self.use_gamma, "weight is applied on rec, but gamma is not initialized"

        # BIAS CONFIGURATION
        self.use_bias = bias_setup is not None
        if self.use_bias:
            bias_cfg = {**DEFAULT_BIAS, **(bias_setup or {})}
            self._setup_vector("bias", bias_cfg)

        # POST-INIT ASSERTIONS FOR SAFETY
        if self.use_gamma or self.use_rec_weight:
            assert self.use_rec_weight, "gamma initialized, but cannot be used as rec_weight is not initialized"
            assert self.use_gamma, "rec_weight initialized, but cannot be used as gamma is not initialized"

        if self.use_pos_scale:
            assert self.use_pos_threshold, "pos_scale initialized, but cannot be used as pos_threshold is not initialized"

        if self.use_neg_scale:
            assert self.use_neg_threshold, "neg_scale initialized, but cannot be used as neg_threshold is not initialized"

        self.zero_states()

    def zero_states(self):
        self.mem = None
        if self.use_alpha:
            self.syn = None
        if self.use_gamma:
            self.rec = None
            self.prev_output = None

    def detach_states(self):
        if self.mem is not None:
            self.mem = self.mem.detach()
        if self.use_alpha:
            if self.syn is not None:
                self.syn = self.syn.detach()
        if self.use_gamma:
            if self.rec is not None:
                self.rec = self.rec.detach()
            if self.prev_output is not None:
                self.prev_output = self.prev_output.detach()

    def _register_tensor(self, name: str, tensor: torch.Tensor, learn: bool):
        if learn:
            setattr(self, name, nn.Parameter(tensor.detach().clone()))
        else:
            self.register_buffer(name, tensor.detach().clone())

    def _setup_decay(self, name, config):
        value, rank, learnable = config["value"], config["rank"], config["learnable"]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                pass
            elif value.ndim == 1:
                assert value.numel() == self.num_neurons, f"{name} decay does not have {self.num_neurons} elements"
            else:
                raise ValueError(f"rank (.ndim) of provided {name} decay is not 0 (scalar) or 1 (vector)")
            decay_tensor = tt_functional.sigmoid_inverse(value)
        else:
            decay = float(value)
            if rank == 0:
                decay_tensor = tt_functional.sigmoid_inverse(torch.tensor(decay))
            elif rank == 1:
                decay_tensor = tt_functional.sigmoid_inverse(torch.full([self.num_neurons], decay))
            else:
                raise ValueError(f"{name} decay rank is not 0 (scalar) or 1 (vector)")
        decay_rank = decay_tensor.ndim
        setattr(self, f"{name}_rank", decay_rank)
        self._register_tensor(f"raw_{name}", decay_tensor, learnable)

    def _setup_threshold(self, name, config):
        value, rank, learnable = config["value"], config["rank"], config["learnable"]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                pass
            elif value.ndim == 1:
                assert value.numel() == self.num_neurons, f"{name} threshold does not have {self.num_neurons} elements"
            else:
                raise ValueError(f"rank (.ndim) of provided {name} threshold is not 0 (scalar) or 1 (vector)")
            threshold_tensor = tt_functional.softplus_inverse(value)
        else:
            threshold = float(value)
            if rank == 0:
                threshold_tensor = tt_functional.softplus_inverse(torch.tensor(threshold))
            elif rank == 1:
                threshold_tensor = tt_functional.softplus_inverse(torch.full([self.num_neurons], threshold))
            else:
                raise ValueError(f"{name} threshold rank is not 0 (scalar) or 1 (vector)")
        threshold_rank = threshold_tensor.ndim
        setattr(self, f"{name}_rank", threshold_rank)
        self._register_tensor(f"raw_{name}", threshold_tensor, learnable)

    def _setup_vector(self, name, config):
        value, rank, learnable = config["value"], config["rank"], config["learnable"]
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                pass
            elif value.ndim == 1:
                assert value.numel() == self.num_neurons, f"{name} does not have {self.num_neurons} elements"
            else:
                raise ValueError(f"rank (.ndim) of provided {name} is not 0 (scalar) or 1 (vector)")
            vector_tensor = value
        else:
            value = float(value)
            if rank == 0:
                vector_tensor = torch.tensor(value)
            elif rank == 1:
                vector_tensor = torch.full([self.num_neurons], value)
            else:
                raise ValueError(f"{name} rank is not 0 (scalar) or 1 (vector)")
        vector_rank = vector_tensor.ndim
        setattr(self, f"{name}_rank", vector_rank)
        self._register_tensor(f"raw_{name}", vector_tensor, learnable)

    @property
    def alpha(self):
        return nn.functional.sigmoid(self.raw_alpha)

    @property
    def beta(self):
        return nn.functional.sigmoid(self.raw_beta)

    @property
    def gamma(self):
        return nn.functional.sigmoid(self.raw_gamma)

    @property
    def pos_threshold(self):
        return nn.functional.softplus(self.raw_pos_threshold)

    @property
    def neg_threshold(self):
        return - nn.functional.softplus(self.raw_neg_threshold)

    @property
    def pos_scale(self):
        return self.raw_pos_scale

    @property
    def neg_scale(self):
        return self.raw_neg_scale

    @property
    def weight(self):
        return self.raw_weight

    @property
    def bias(self):
        return self.raw_bias

    def forward(self, x: torch.Tensor):
        # INITIALIZATION
        x_moved = x.movedim(self.dim, -1)
        # we use beta, mem needs to be initialized
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        mem_moved = self.mem.movedim(self.dim, -1)
        mem_delta = torch.zeros_like(mem_moved)

        # if we have alpha, syn needs to be initialized, and x_moved needs to be added to the correct delta
        if self.use_alpha:
            if self.syn is None:
                self.syn = torch.zeros_like(x)
            syn_moved = self.syn.movedim(self.dim, -1)
            syn_delta = x_moved
        else:
            mem_delta = mem_delta + x_moved

        # if we have gamma, rec and prev_output needs to be initialized
        if self.use_gamma:
            if self.rec is None:
                self.rec = torch.zeros_like(x)
            if self.prev_output is None:
                self.prev_output = torch.zeros_like(x)

            rec_moved = self.rec.movedim(self.dim, -1)
            rec_delta = self.prev_output.movedim(self.dim, -1)
            if self.rec_is_ema:
                rec_delta = rec_delta * (1 - self.gamma)
            rec_moved = rec_moved * self.gamma + rec_delta
            mem_delta = mem_delta + rec_moved * self.rec_weight
            self.rec = rec_moved.movedim(-1, self.dim)

        # if we use bias, let's quickly add it to the mem_delta
        if self.use_bias:
            mem_delta = mem_delta + self.bias

        # ACTUAL CALCULATION LOGIC
        if self.use_alpha:
            if self.syn_is_ema:
                syn_delta = syn_delta * (1 - self.alpha)
            syn_moved = syn_moved * self.alpha + syn_delta
            mem_delta = mem_delta + syn_moved
            self.syn = syn_moved.movedim(-1, self.dim)

        if self.mem_is_ema:
            mem_delta = mem_delta * (1 - self.beta)

        mem_moved = mem_moved * self.beta + mem_delta

        if self.use_pos_threshold or self.use_neg_threshold:
            output_moved = torch.zeros_like(mem_moved)
            if self.use_pos_threshold:
                pos_spikes = self.pos_surrogate(mem_moved - self.pos_threshold)
                mem_moved = mem_moved - pos_spikes * self.pos_threshold
                if self.use_pos_scale:
                    pos_spikes = pos_spikes * self.pos_scale
                output_moved = output_moved + pos_spikes
            if self.use_neg_threshold:
                neg_spikes = -self.neg_surrogate(self.neg_threshold - mem_moved)
                mem_moved = mem_moved + neg_spikes * self.neg_threshold  # both are negative, create positive delta
                if self.use_neg_scale:
                    neg_spikes = neg_spikes * self.neg_scale
                output_moved = output_moved + neg_spikes
        else:
            output_moved = mem_moved

        output = output_moved.movedim(-1, self.dim)
        self.mem = mem_moved.movedim(-1, self.dim)

        # If we use recurrence, we need to save prev_output
        if self.use_gamma:
            self.prev_output = output

        return output
