from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from ._snnlayer import Layer as SNNLayer
from . import spike_fn as spike_functions


class LIT(SNNLayer):
    r"""A leaky integrate-and-ternary-fire layer.

    ``LIT`` stores one membrane trace and can emit positive, zero, or negative
    output. Positive firing is controlled by ``pos_threshold``; negative firing
    is controlled by ``neg_threshold``. The negative branch returns negative
    values by convention.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay, constrained
            to ``(0, 1)``.
        pos_threshold (float or torch.Tensor, default=1.0): positive firing
            threshold, constrained to positive values.
        neg_threshold (float or torch.Tensor, default=1.0): magnitude of the
            negative firing threshold, constrained to positive values.
            boundaries.
        dim (int, default=-1): the dimension along which the layer operates.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        mem: membrane state.
        beta: activated membrane decay.
        pos_threshold: activated positive threshold.
        neg_threshold: activated negative threshold magnitude.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        With the default ``spike_fn=tt.snn.spike_fn.deterministic``, positive and negative
        outputs are hard firing events with opposite signs and surrogate gradients. Pseudocode looks
        as follows:

        ::

            mem = beta * mem + x
            return pos + neg

    Examples::

        >>> layer = tt.snn.LIT(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("mem")
        self.define_state("prev_output")
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        mem = self.to_working_dim(self.mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        mem = mem * self.beta + x

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.prev_output = spikes
        self.mem = self.from_working_dim(mem)

        return spikes


class DLIT(SNNLayer):
    r"""A dual leaky integrate-and-ternary-fire layer.

    ``DLIT`` splits membrane integration into positive and negative branches and
    emits ternary-style output. The summed membrane is thresholded, and each
    reset is split evenly across the positive and negative membrane branches.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        dim (int, default=-1): the dimension along which the layer operates.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            pos_mem = pos_beta * pos_mem + where(x >= 0, x, 0)
            neg_mem = neg_beta * neg_mem + where(x < 0, x, 0)
            mem = pos_mem + neg_mem
            return pos + neg

    Examples::

        >>> layer = tt.snn.DLIT(num_neurons=32)
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
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("pos_mem")
        self.define_state("neg_mem")
        self.define_state("prev_output")
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        pos_mem = self.to_working_dim(self.pos_mem)
        neg_mem = self.to_working_dim(self.neg_mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_mem = pos_mem * self.pos_beta + torch.where(x >= 0, x, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(x < 0, x, 0.0)

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.prev_output = spikes
        self.pos_mem = self.from_working_dim(pos_mem)
        self.neg_mem = self.from_working_dim(neg_mem)

        return spikes


class SLIT(SNNLayer):
    r"""A synaptic leaky integrate-and-ternary-fire layer.

    ``SLIT`` smooths the input through a synaptic trace before membrane
    integration and ternary firing. It is the ternary counterpart of ``SLIB``.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        dim (int, default=-1): the dimension along which the layer operates.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

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
            return pos + neg

    Examples::

        >>> layer = tt.snn.SLIT(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_neurons: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)
        self.define_state("syn")
        self.define_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.define_state("mem")
        self.define_state("prev_output")
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        mem = self.to_working_dim(self.mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        syn = self.to_working_dim(self.syn)
        syn = syn * self.alpha + x * (1 - self.alpha)

        mem = mem * self.beta + syn

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.prev_output = spikes
        self.syn = self.from_working_dim(syn)
        self.mem = self.from_working_dim(mem)

        return spikes


class RLIT(SNNLayer):
    r"""A recurrent leaky integrate-and-ternary-fire layer.

    ``RLIT`` adds a recurrent trace of the previous ternary output. The recurrent
    trace is scaled by ``rec_weight`` and added to the input before membrane
    integration and ternary firing.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        gamma (float or torch.Tensor, default=0.9): recurrent trace decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        rec_weight (float or torch.Tensor, default=0.0): recurrent input scale.
        dim (int, default=-1): the dimension along which the layer operates.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        gamma_rank (Literal[0, 1], default=1): scalar or per-neuron recurrent
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            recurrent scale.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_gamma (bool, default=True): whether ``gamma`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_rec_weight (bool, default=True): whether ``rec_weight`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

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
            prev_output = pos + neg
            return prev_output

    Examples::

        >>> layer = tt.snn.RLIT(num_neurons=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_neurons: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("mem")
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.define_state("rec")
        self.define_state("prev_output")
        self.define_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        mem = self.to_working_dim(self.mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        rec = self.to_working_dim(self.rec)
        rec = rec * self.gamma + prev_output * (1 - self.gamma)

        mem_delta = rec * self.rec_weight + x

        mem = mem * self.beta + mem_delta

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.mem = self.from_working_dim(mem)
        self.rec = self.from_working_dim(rec)
        self.prev_output = spikes

        return spikes


class DSLIT(SNNLayer):
    r"""A dual synaptic leaky integrate-and-ternary-fire layer.

    ``DSLIT`` combines dual positive/negative traces with a synaptic stage and
    ternary output. It smooths positive and negative input separately before
    integrating the combined synaptic current into dual membrane traces.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        dim (int, default=-1): the dimension along which the layer operates.
        pos_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            synaptic decay.
        neg_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            synaptic decay.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            membrane decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            membrane decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        learn_pos_alpha (bool, default=True): whether ``pos_alpha`` is trainable.
        learn_neg_alpha (bool, default=True): whether ``neg_alpha`` is trainable.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        pos_syn: positive synaptic state.
        neg_syn: negative synaptic state.
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where ``num_neurons``
          is at index ``dim``.
        - **Output**: tensor with the same shape as the input.

        The membrane reset is split evenly across the two membrane branches.

    Examples::

        >>> layer = tt.snn.DSLIT(num_neurons=32)
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
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("pos_syn")
        self.define_state("neg_syn")
        self.define_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self.define_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self.define_state("pos_mem")
        self.define_state("neg_mem")
        self.define_state("prev_output")
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        pos_mem = self.to_working_dim(self.pos_mem)
        neg_mem = self.to_working_dim(self.neg_mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_syn = self.to_working_dim(self.pos_syn)
        neg_syn = self.to_working_dim(self.neg_syn)
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = self.from_working_dim(pos_syn)
        self.neg_syn = self.from_working_dim(neg_syn)

        syn = pos_syn + neg_syn

        pos_mem = pos_mem * self.pos_beta + torch.where(syn >= 0, syn, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(syn <= 0, syn, 0.0)

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.prev_output = spikes
        self.pos_mem = self.from_working_dim(pos_mem)
        self.neg_mem = self.from_working_dim(neg_mem)

        return spikes


class DRLIT(SNNLayer):
    r"""A dual recurrent leaky integrate-and-ternary-fire layer.

    ``DRLIT`` keeps dual membrane traces and dual recurrent traces for ternary
    output. The previous output is split by sign into recurrent branches, then
    reintegrated with the current input before the ternary firing decision.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_gamma (float or torch.Tensor, default=0.9): positive recurrent decay.
        neg_gamma (float or torch.Tensor, default=0.9): negative recurrent decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
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
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        pos_rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive recurrent scale.
        neg_rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative recurrent scale.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_gamma (bool, default=True): whether ``pos_gamma`` is trainable.
        learn_neg_gamma (bool, default=True): whether ``neg_gamma`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_rec_weight (bool, default=True): whether ``pos_rec_weight`` is
            trainable.
        learn_neg_rec_weight (bool, default=True): whether ``neg_rec_weight`` is
            trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

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

        ``DRLIT`` is the ternary recurrent layer to reach for when positive and
        negative recurrent history should use different decays and gains.

    Examples::

        >>> layer = tt.snn.DRLIT(num_neurons=32)
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
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("pos_mem")
        self.define_state("neg_mem")
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.define_state("pos_rec")
        self.define_state("neg_rec")
        self.define_state("prev_output")
        self.define_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self.define_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self.define_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        pos_mem = self.to_working_dim(self.pos_mem)
        neg_mem = self.to_working_dim(self.neg_mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_rec = self.to_working_dim(self.pos_rec)
        neg_rec = self.to_working_dim(self.neg_rec)

        pos_rec = pos_rec * self.pos_gamma + torch.where(prev_output >= 0, prev_output, 0.0) * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + torch.where(prev_output <= 0, prev_output, 0.0) * (1 - self.neg_gamma)

        self.pos_rec = self.from_working_dim(pos_rec)
        self.neg_rec = self.from_working_dim(neg_rec)

        rec = pos_rec + neg_rec

        pos_mem_delta = torch.where(rec >= 0, rec, 0.0) * self.pos_rec_weight + torch.where(x >= 0, x, 0.0)
        neg_mem_delta = torch.where(rec <= 0, rec, 0.0) * self.neg_rec_weight + torch.where(x <= 0, x, 0.0)

        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.pos_mem = self.from_working_dim(pos_mem)
        self.neg_mem = self.from_working_dim(neg_mem)
        self.prev_output = spikes

        return spikes


class SRLIT(SNNLayer):
    r"""A synaptic recurrent leaky integrate-and-ternary-fire layer.

    ``SRLIT`` combines a synaptic input trace with a recurrent trace of the
    previous ternary output. The membrane receives both the smoothed input and
    the recurrent current before the ternary firing decision.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        gamma (float or torch.Tensor, default=0.9): recurrent trace decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        rec_weight (float or torch.Tensor, default=0.0): recurrent input scale.
        dim (int, default=-1): the dimension along which the layer operates.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        gamma_rank (Literal[0, 1], default=1): scalar or per-neuron recurrent
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            recurrent scale.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_gamma (bool, default=True): whether ``gamma`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_rec_weight (bool, default=True): whether ``rec_weight`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

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
            prev_output = pos + neg
            return prev_output

    Examples::

        >>> layer = tt.snn.SRLIT(num_neurons=32)
        >>> input = torch.randn(16, 32)
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
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("syn")
        self.define_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.define_state("mem")
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.define_state("rec")
        self.define_state("prev_output")
        self.define_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        mem = self.to_working_dim(self.mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        syn = self.to_working_dim(self.syn)
        syn = syn * self.alpha + x * (1 - self.alpha)

        rec = self.to_working_dim(self.rec)
        rec = rec * self.gamma + prev_output * (1 - self.gamma)
        self.rec = self.from_working_dim(rec)

        mem_delta = rec * self.rec_weight + syn

        mem = mem * self.beta + mem_delta

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.syn = self.from_working_dim(syn)
        self.mem = self.from_working_dim(mem)
        self.prev_output = spikes

        return spikes


class DSRLIT(SNNLayer):
    r"""A dual synaptic recurrent leaky integrate-and-ternary-fire layer.

    ``DSRLIT`` combines every ternary trace mechanism: dual synaptic traces,
    dual recurrent traces, dual membrane traces, and positive/negative firing
    thresholds. It is the most expressive unscaled ternary SNN layer.

    Args:
        num_neurons (int): number of neurons in the target dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_gamma (float or torch.Tensor, default=0.9): positive recurrent decay.
        neg_gamma (float or torch.Tensor, default=0.9): negative recurrent decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
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
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
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
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_rec_weight (bool, default=True): whether ``pos_rec_weight`` is
            trainable.
        learn_neg_rec_weight (bool, default=True): whether ``neg_rec_weight`` is
            trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

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

        Use ``DSRLIT`` when both input history and output history should retain
        sign-specific dynamics before ternary firing.

    Examples::

        >>> layer = tt.snn.DSRLIT(num_neurons=32)
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
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            dim: int = -1,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__(num_neurons, dim)

        self.define_state("pos_syn")
        self.define_state("neg_syn")
        self.define_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self.define_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self.define_state("pos_mem")
        self.define_state("neg_mem")
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.define_state("pos_rec")
        self.define_state("neg_rec")
        self.define_state("prev_output")
        self.define_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self.define_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self.define_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)
        x = self.to_working_dim(x)

        pos_mem = self.to_working_dim(self.pos_mem)
        neg_mem = self.to_working_dim(self.neg_mem)
        prev_output = self.to_working_dim(self.prev_output)
        pos_prev_output = torch.where(prev_output >= 0, prev_output, 0.0)
        neg_prev_output = torch.where(prev_output <= 0, prev_output, 0.0)
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_syn = self.to_working_dim(self.pos_syn)
        neg_syn = self.to_working_dim(self.neg_syn)
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = self.from_working_dim(pos_syn)
        self.neg_syn = self.from_working_dim(neg_syn)

        syn = pos_syn + neg_syn

        pos_rec = self.to_working_dim(self.pos_rec)
        neg_rec = self.to_working_dim(self.neg_rec)

        pos_rec = pos_rec * self.pos_gamma + torch.where(prev_output >= 0, prev_output, 0.0) * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + torch.where(prev_output <= 0, prev_output, 0.0) * (1 - self.neg_gamma)

        self.pos_rec = self.from_working_dim(pos_rec)
        self.neg_rec = self.from_working_dim(neg_rec)

        rec = pos_rec + neg_rec

        pos_mem_delta = torch.where(rec >= 0, rec, 0.0) * self.pos_rec_weight + torch.where(syn >= 0, syn, 0.0)
        neg_mem_delta = torch.where(rec <= 0, rec, 0.0) * self.neg_rec_weight + torch.where(syn <= 0, syn, 0.0)

        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        spikes = pos_spikes + neg_spikes

        spikes = self.from_working_dim(spikes)
        self.pos_mem = self.from_working_dim(pos_mem)
        self.neg_mem = self.from_working_dim(neg_mem)
        self.prev_output = spikes

        return spikes
