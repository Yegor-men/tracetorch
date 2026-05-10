from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ._snnlayer import Layer as SNNLayer
from .. import functional


class LIB(SNNLayer):
    r"""A leaky integrate-and-binary-fire layer.

    ``LIB`` is traceTorch's one-sided firing layer. It stores one membrane trace,
    converts the distance from threshold into a firing probability, optionally
    quantizes that probability, subtracts the threshold-scaled output from the
    membrane, and returns the output.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay, constrained
            to ``(0, 1)``.
        threshold (float or torch.Tensor, default=1.0): positive firing
            threshold, constrained to positive values.
        bias (float or torch.Tensor, default=0.0): additive bias applied before
            the spike function.
        dim (int, default=-1): the dimension along which the layer operates.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): function that maps
            membrane distance from threshold to a firing probability.
        quant_fn (Callable, default=nn.Identity()): function that maps firing
            probability to the returned spike value.

    Attributes:
        mem: membrane state.
        beta: activated membrane decay.
        threshold: activated positive threshold.
        bias: activated bias.
        spike_fn: spike probability function.
        quant_fn: output quantization function.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        With the default ``quant_fn=nn.Identity()``, the layer returns smooth
        firing probabilities. Pass a straight-through quantizer such as
        ``tt.functional.round_ste()`` for harder binary events. Pseudocode looks
        as follows:

        ::

            mem = beta * mem + x
            spike_prob = spike_fn(mem - threshold + bias)
            spikes = quant_fn(spike_prob)
            mem = mem - spikes * threshold
            return spikes

    Examples::

        >>> layer = tt.snn.LIB(num_neurons=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn
        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + x

        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        mem = mem - spikes * self.threshold

        spikes = self._from_working_dim(spikes)
        self.mem = self._from_working_dim(mem)

        return spikes


class DLIB(SNNLayer):
    r"""A dual leaky integrate-and-binary-fire layer.

    ``DLIB`` splits membrane integration into positive and negative branches, but
    still emits a one-sided binary-style output. The two membrane branches are
    summed for thresholding, then the reset is split evenly across both branches.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        dim (int, default=-1): the dimension along which the layer operates.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.
        threshold: activated positive threshold.
        bias: activated bias.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            pos_mem = pos_beta * pos_mem + where(x >= 0, x, 0)
            neg_mem = neg_beta * neg_mem + where(x <= 0, x, 0)
            mem = pos_mem + neg_mem
            spikes = quant_fn(spike_fn(mem - threshold + bias))
            pos_mem = pos_mem - 0.5 * spikes * threshold
            neg_mem = neg_mem - 0.5 * spikes * threshold
            return spikes

    Examples::

        >>> layer = tt.snn.DLIB(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + torch.where(x >= 0, x, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(x <= 0, x, 0.0)

        mem = pos_mem + neg_mem

        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        pos_mem = pos_mem - spikes * self.threshold * 0.5
        neg_mem = neg_mem - spikes * self.threshold * 0.5

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)

        return spikes


class SLIB(SNNLayer):
    r"""A synaptic leaky integrate-and-binary-fire layer.

    ``SLIB`` smooths the input through a synaptic trace before membrane
    integration and one-sided firing. This is useful when the input should behave
    like a current with its own time constant rather than an instantaneous
    membrane increment.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        dim (int, default=-1): the dimension along which the layer operates.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        syn: synaptic state.
        mem: membrane state.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            syn = alpha * syn + (1 - alpha) * x
            mem = beta * mem + syn
            spikes = quant_fn(spike_fn(mem - threshold + bias))
            mem = mem - spikes * threshold
            return spikes

    Examples::

        >>> layer = tt.snn.SLIB(num_neurons=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("syn")
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        syn = self._to_working_dim(self.syn)
        syn = syn * self.alpha + x * (1 - self.alpha)

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + syn
        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        mem = mem - spikes * self.threshold

        spikes = self._from_working_dim(spikes)
        self.syn = self._from_working_dim(syn)
        self.mem = self._from_working_dim(mem)

        return spikes


class RLIB(SNNLayer):
    r"""A recurrent leaky integrate-and-binary-fire layer.

    ``RLIB`` adds a recurrent trace of the previous output. The recurrent trace
    is decayed with ``gamma``, scaled by ``rec_weight``, and added to the current
    input before membrane integration.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        gamma (float or torch.Tensor, default=0.9): recurrent trace decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        rec_weight (float or torch.Tensor, default=0.0): recurrent input scale.
        dim (int, default=-1): the dimension along which the layer operates.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        gamma_rank (Literal[0, 1], default=1): scalar or per-neuron recurrent
            decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            recurrent scale.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_gamma (bool, default=True): whether ``gamma`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        learn_rec_weight (bool, default=True): whether ``rec_weight`` is trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        mem: membrane state.
        rec: recurrent trace state.
        prev_output: previous returned output.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            rec = gamma * rec + (1 - gamma) * prev_output
            mem = beta * mem + x + rec_weight * rec
            spikes = quant_fn(spike_fn(mem - threshold + bias))
            mem = mem - spikes * threshold
            prev_output = spikes
            return spikes

    Examples::

        >>> layer = tt.snn.RLIB(num_neurons=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self._initialize_state("rec")
        self._initialize_state("prev_output")
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        rec = self._to_working_dim(self.rec)
        prev_output = self._to_working_dim(self.prev_output)
        rec = rec * self.gamma + prev_output * (1 - self.gamma)
        self.rec = self._from_working_dim(rec)

        mem_delta = rec * self.rec_weight + x

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + mem_delta

        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        mem = mem - spikes * self.threshold

        spikes = self._from_working_dim(spikes)
        self.mem = self._from_working_dim(mem)
        self.prev_output = spikes

        return spikes


class DSLIB(SNNLayer):
    r"""A dual synaptic leaky integrate-and-binary-fire layer.

    ``DSLIB`` combines dual positive/negative traces with a synaptic stage and a
    one-sided firing output. Positive and negative inputs are smoothed
    separately, summed, integrated into dual membrane traces, and thresholded as
    one combined membrane.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        dim (int, default=-1): the dimension along which the layer operates.
        pos_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            synaptic decay.
        neg_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            synaptic decay.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            membrane decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            membrane decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        learn_pos_alpha (bool, default=True): whether ``pos_alpha`` is trainable.
        learn_neg_alpha (bool, default=True): whether ``neg_alpha`` is trainable.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        pos_syn: positive synaptic state.
        neg_syn: negative synaptic state.
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        The reset is split evenly across the dual membrane branches. Pseudocode
        looks as follows:

        ::

            pos_syn = pos_alpha * pos_syn + (1 - pos_alpha) * where(x >= 0, x, 0)
            neg_syn = neg_alpha * neg_syn + (1 - neg_alpha) * where(x <= 0, x, 0)
            syn = pos_syn + neg_syn
            pos_mem = pos_beta * pos_mem + where(syn >= 0, syn, 0)
            neg_mem = neg_beta * neg_mem + where(syn <= 0, syn, 0)
            spikes = quant_fn(spike_fn(pos_mem + neg_mem - threshold + bias))
            pos_mem = pos_mem - 0.5 * spikes * threshold
            neg_mem = neg_mem - 0.5 * spikes * threshold
            return spikes

    Examples::

        >>> layer = tt.snn.DSLIB(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_syn")
        self._initialize_state("neg_syn")
        self._register_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self._register_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_syn = self._to_working_dim(self.pos_syn)
        neg_syn = self._to_working_dim(self.neg_syn)
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = self._from_working_dim(pos_syn)
        self.neg_syn = self._from_working_dim(neg_syn)

        syn = pos_syn + neg_syn

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + torch.where(syn >= 0, syn, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(syn <= 0, syn, 0.0)

        mem = pos_mem + neg_mem

        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        pos_mem = pos_mem - spikes * self.threshold * 0.5
        neg_mem = neg_mem - spikes * self.threshold * 0.5

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)

        return spikes


class DRLIB(SNNLayer):
    r"""A dual recurrent leaky integrate-and-binary-fire layer.

    ``DRLIB`` keeps positive and negative membrane traces and positive and
    negative recurrent traces. The previous output is split by sign, smoothed
    into recurrent traces, scaled, and integrated with the current input before a
    one-sided binary firing decision.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_gamma (float or torch.Tensor, default=0.9): positive recurrent decay.
        neg_gamma (float or torch.Tensor, default=0.9): negative recurrent decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        pos_rec_weight (float or torch.Tensor, default=0.0): positive recurrent
            input scale.
        neg_rec_weight (float or torch.Tensor, default=0.0): negative recurrent
            input scale.
        dim (int, default=-1): the dimension along which the layer operates.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            membrane decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            membrane decay.
        pos_gamma_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            recurrent decay.
        neg_gamma_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            recurrent decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        pos_rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive recurrent scale.
        neg_rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative recurrent scale.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_gamma (bool, default=True): whether ``pos_gamma`` is trainable.
        learn_neg_gamma (bool, default=True): whether ``neg_gamma`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        learn_pos_rec_weight (bool, default=True): whether ``pos_rec_weight`` is
            trainable.
        learn_neg_rec_weight (bool, default=True): whether ``neg_rec_weight`` is
            trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.
        pos_rec: positive recurrent trace state.
        neg_rec: negative recurrent trace state.
        prev_output: previous returned output.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            pos_rec = pos_gamma * pos_rec + (1 - pos_gamma) * where(prev_output >= 0, prev_output, 0)
            neg_rec = neg_gamma * neg_rec + (1 - neg_gamma) * where(prev_output <= 0, prev_output, 0)
            pos_mem = pos_beta * pos_mem + pos_rec_weight * where(pos_rec + neg_rec >= 0, pos_rec + neg_rec, 0) + where(x >= 0, x, 0)
            neg_mem = neg_beta * neg_mem + neg_rec_weight * where(pos_rec + neg_rec <= 0, pos_rec + neg_rec, 0) + where(x <= 0, x, 0)
            spikes = quant_fn(spike_fn(pos_mem + neg_mem - threshold + bias))
            prev_output = spikes
            return spikes

    Examples::

        >>> layer = tt.snn.DRLIB(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self._initialize_state("pos_rec")
        self._initialize_state("neg_rec")
        self._initialize_state("prev_output")
        self._register_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self._register_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

        self._register_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self._register_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_rec = self._to_working_dim(self.pos_rec)
        neg_rec = self._to_working_dim(self.neg_rec)
        prev_output = self._to_working_dim(self.prev_output)

        pos_rec = pos_rec * self.pos_gamma + torch.where(prev_output >= 0, prev_output, 0.0) * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + torch.where(prev_output <= 0, prev_output, 0.0) * (1 - self.neg_gamma)

        self.pos_rec = self._from_working_dim(pos_rec)
        self.neg_rec = self._from_working_dim(neg_rec)

        rec = pos_rec + neg_rec

        pos_mem_delta = torch.where(rec >= 0, rec, 0.0) * self.pos_rec_weight + torch.where(x >= 0, x, 0.0)
        neg_mem_delta = torch.where(rec <= 0, rec, 0.0) * self.neg_rec_weight + torch.where(x <= 0, x, 0.0)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        pos_mem = pos_mem - spikes * self.threshold * 0.5
        neg_mem = neg_mem - spikes * self.threshold * 0.5

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)
        self.prev_output = spikes

        return spikes


class SRLIB(SNNLayer):
    r"""A synaptic recurrent leaky integrate-and-binary-fire layer.

    ``SRLIB`` combines an input synaptic trace with a recurrent trace of the
    previous output. The membrane receives both the smoothed input and the scaled
    recurrent current before one-sided firing.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        gamma (float or torch.Tensor, default=0.9): recurrent trace decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        rec_weight (float or torch.Tensor, default=0.0): recurrent input scale.
        dim (int, default=-1): the dimension along which the layer operates.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        gamma_rank (Literal[0, 1], default=1): scalar or per-neuron recurrent
            decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            recurrent scale.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_gamma (bool, default=True): whether ``gamma`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        learn_rec_weight (bool, default=True): whether ``rec_weight`` is trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        syn: synaptic state.
        mem: membrane state.
        rec: recurrent trace state.
        prev_output: previous returned output.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            syn = alpha * syn + (1 - alpha) * x
            rec = gamma * rec + (1 - gamma) * prev_output
            mem = beta * mem + syn + rec_weight * rec
            spikes = quant_fn(spike_fn(mem - threshold + bias))
            mem = mem - spikes * threshold
            prev_output = spikes
            return spikes

    Examples::

        >>> layer = tt.snn.SRLIB(num_neurons=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            threshold: Union[float, torch.Tensor] = 1.0,
            bias: Union[float, torch.Tensor] = 0.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            threshold_rank: Literal[0, 1] = 1,
            bias_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("syn")
        self._register_decay("alpha", alpha, alpha_rank, learn_alpha)

        self._initialize_state("mem")
        self._register_decay("beta", beta, beta_rank, learn_beta)

        self._initialize_state("rec")
        self._initialize_state("prev_output")
        self._register_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

        self._register_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        syn = self._to_working_dim(self.syn)
        syn = syn * self.alpha + x * (1 - self.alpha)
        self.syn = self._from_working_dim(syn)

        rec = self._to_working_dim(self.rec)
        prev_output = self._to_working_dim(self.prev_output)
        rec = rec * self.gamma + prev_output * (1 - self.gamma)
        self.rec = self._from_working_dim(rec)

        mem_delta = rec * self.rec_weight + syn

        mem = self._to_working_dim(self.mem)
        mem = mem * self.beta + mem_delta
        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        mem = mem - spikes * self.threshold

        spikes = self._from_working_dim(spikes)
        self.mem = self._from_working_dim(mem)
        self.prev_output = spikes

        return spikes


class DSRLIB(SNNLayer):
    r"""A dual synaptic recurrent leaky integrate-and-binary-fire layer.

    ``DSRLIB`` is the most expressive binary SNN layer in traceTorch. It combines
    dual positive/negative synaptic traces, dual positive/negative recurrent
    traces, dual positive/negative membrane traces, and a one-sided binary firing
    output.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_gamma (float or torch.Tensor, default=0.9): positive recurrent decay.
        neg_gamma (float or torch.Tensor, default=0.9): negative recurrent decay.
        threshold (float or torch.Tensor, default=1.0): positive firing threshold.
        bias (float or torch.Tensor, default=0.0): additive bias before firing.
        pos_rec_weight (float or torch.Tensor, default=0.0): positive recurrent
            input scale.
        neg_rec_weight (float or torch.Tensor, default=0.0): negative recurrent
            input scale.
        dim (int, default=-1): the dimension along which the layer operates.
        pos_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            synaptic decay.
        neg_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            synaptic decay.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            membrane decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            membrane decay.
        pos_gamma_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            recurrent decay.
        neg_gamma_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            recurrent decay.
        threshold_rank (Literal[0, 1], default=1): scalar or per-neuron threshold.
        bias_rank (Literal[0, 1], default=1): scalar or per-neuron bias.
        pos_rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive recurrent scale.
        neg_rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative recurrent scale.
        learn_pos_alpha (bool, default=True): whether ``pos_alpha`` is trainable.
        learn_neg_alpha (bool, default=True): whether ``neg_alpha`` is trainable.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_gamma (bool, default=True): whether ``pos_gamma`` is trainable.
        learn_neg_gamma (bool, default=True): whether ``neg_gamma`` is trainable.
        learn_threshold (bool, default=True): whether ``threshold`` is trainable.
        learn_bias (bool, default=True): whether ``bias`` is trainable.
        learn_pos_rec_weight (bool, default=True): whether ``pos_rec_weight`` is
            trainable.
        learn_neg_rec_weight (bool, default=True): whether ``neg_rec_weight`` is
            trainable.
        spike_fn (Callable, default=tt.functional.sigmoid4x): spike probability
            function.
        quant_fn (Callable, default=nn.Identity()): output quantization function.

    Attributes:
        pos_syn: positive synaptic state.
        neg_syn: negative synaptic state.
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.
        pos_rec: positive recurrent trace state.
        neg_rec: negative recurrent trace state.
        prev_output: previous returned output.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        ``DSRLIB`` is useful when the sign of the input and the sign of the
        recurrent history should both have independent memory. The firing output
        is still one-sided: negative internal evidence can suppress firing, but
        the returned output is non-negative unless a custom ``quant_fn`` changes
        that convention.

    Examples::

        >>> layer = tt.snn.DSRLIB(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """
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
            bias: Union[float, torch.Tensor] = 0.0,
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
            bias_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_threshold: bool = True,
            learn_bias: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=functional.sigmoid4x,
            quant_fn=nn.Identity(),
    ):
        super().__init__(num_neurons, dim)

        self._initialize_state("pos_syn")
        self._initialize_state("neg_syn")
        self._register_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self._register_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self._initialize_state("pos_mem")
        self._initialize_state("neg_mem")
        self._register_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self._register_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self._initialize_state("pos_rec")
        self._initialize_state("neg_rec")
        self._initialize_state("prev_output")
        self._register_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self._register_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.quant_fn = quant_fn

        self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
        self._register_bias("bias", bias, bias_rank, learn_bias)

        self._register_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self._register_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self._ensure_states(x)
        x = self._to_working_dim(x)

        pos_syn = self._to_working_dim(self.pos_syn)
        neg_syn = self._to_working_dim(self.neg_syn)
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = self._from_working_dim(pos_syn)
        self.neg_syn = self._from_working_dim(neg_syn)

        syn = pos_syn + neg_syn

        pos_rec = self._to_working_dim(self.pos_rec)
        neg_rec = self._to_working_dim(self.neg_rec)
        prev_output = self._to_working_dim(self.prev_output)

        pos_rec = pos_rec * self.pos_gamma + torch.where(prev_output >= 0, prev_output, 0.0) * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + torch.where(prev_output <= 0, prev_output, 0.0) * (1 - self.neg_gamma)

        self.pos_rec = self._from_working_dim(pos_rec)
        self.neg_rec = self._from_working_dim(neg_rec)

        rec = pos_rec + neg_rec

        pos_mem_delta = torch.where(rec >= 0, rec, 0.0) * self.pos_rec_weight + torch.where(syn >= 0, syn, 0.0)
        neg_mem_delta = torch.where(rec <= 0, rec, 0.0) * self.neg_rec_weight + torch.where(syn <= 0, syn, 0.0)

        pos_mem = self._to_working_dim(self.pos_mem)
        neg_mem = self._to_working_dim(self.neg_mem)
        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        spike_prob = self.spike_fn(mem - self.threshold + self.bias)
        spikes = self.quant_fn(spike_prob)

        pos_mem = pos_mem - spikes * self.threshold * 0.5
        neg_mem = neg_mem - spikes * self.threshold * 0.5

        spikes = self._from_working_dim(spikes)
        self.pos_mem = self._from_working_dim(pos_mem)
        self.neg_mem = self._from_working_dim(neg_mem)
        self.prev_output = spikes

        return spikes
