from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from ._snnlayer import Layer as SNNLayer
from . import spike_fn as spike_functions


class LITS(SNNLayer):
    r"""A leaky integrate-and-ternary-scaled-fire layer.

    ``LITS`` is the scaled-output variant of ``LIT``. It computes positive and
    negative ternary firing decisions, resets the membrane with the unscaled
    spike signs, then multiplies positive and negative outputs by separate
    learnable or fixed scales.

    Args:
        num_features (int): number of neurons in the final dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay, constrained
            to ``(0, 1)``.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): multiplier applied to
            positive output events.
        neg_scale (float or torch.Tensor, default=1.0): multiplier applied to
            negative output events.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        mem: membrane state.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            mem = beta * mem + x
            return pos * pos_scale + neg * neg_scale

    Examples::

        >>> layer = tt.snn.LITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("mem", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        mem = self.mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        mem = mem * self.beta + x

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.mem = mem

        return scaled_spikes


class DLITS(SNNLayer):
    r"""A dual leaky integrate-and-ternary-scaled-fire layer.

    ``DLITS`` combines dual membrane traces with scaled ternary output. The
    summed membrane is thresholded, the reset is split across the dual traces,
    and the returned positive and negative outputs are scaled independently.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        The output scales affect what downstream layers receive; membrane reset
        still uses the unscaled threshold crossing events.

    Examples::

        >>> layer = tt.snn.DLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("pos_mem", (num_features,))
        self.define_state("neg_mem", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn

        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_mem = pos_mem * self.pos_beta + torch.where(x >= 0, x, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(x <= 0, x, 0.0)

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        return scaled_spikes


class SLITS(SNNLayer):
    r"""A synaptic leaky integrate-and-ternary-scaled-fire layer.

    ``SLITS`` smooths the input with a synaptic trace before membrane integration
    and scaled ternary output.

    Args:
        num_features (int): number of neurons in the final dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        syn: synaptic state.
        mem: membrane state.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            syn = alpha * syn + (1 - alpha) * x
            mem = beta * mem + syn
            return pos * pos_scale + neg * neg_scale

    Examples::

        >>> layer = tt.snn.SLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("syn", (num_features,))
        self.define_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.define_state("mem", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.spike_fn = spike_fn

        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        mem = self.mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        syn = self.syn
        syn = syn * self.alpha + x * (1 - self.alpha)

        mem = mem * self.beta + syn

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.syn = syn
        self.mem = mem

        return scaled_spikes


class RLITS(SNNLayer):
    r"""A recurrent leaky integrate-and-ternary-scaled-fire layer.

    ``RLITS`` adds a recurrent trace of the previous scaled ternary output. The
    recurrent trace is scaled by ``rec_weight`` and added before membrane
    integration; the current output is then scaled separately for positive and
    negative events.

    Args:
        num_features (int): number of neurons in the final dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        gamma (float or torch.Tensor, default=0.9): recurrent trace decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
        rec_weight (float or torch.Tensor, default=0.0): recurrent input scale.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        gamma_rank (Literal[0, 1], default=1): scalar or per-neuron recurrent
            decay.
        pos_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            positive threshold.
        neg_threshold_rank (Literal[0, 1], default=1): scalar or per-neuron
            negative threshold magnitude.
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
        rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            recurrent scale.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_gamma (bool, default=True): whether ``gamma`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
        learn_rec_weight (bool, default=True): whether ``rec_weight`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        mem: membrane state.
        rec: recurrent trace state.
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        ``pos_prev_output`` and ``neg_prev_output`` retain unscaled firing
        outputs. Output scales do not affect reset or recurrent history.

    Examples::

        >>> layer = tt.snn.RLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("mem", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.define_state("rec", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn

        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self.define_unbound_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        mem = self.mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        rec = self.rec
        rec = rec * self.gamma + (pos_prev_output + neg_prev_output) * (1 - self.gamma)

        mem_delta = rec * self.rec_weight + x

        mem = mem * self.beta + mem_delta

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.rec = rec
        self.mem = mem

        return scaled_spikes


class DSLITS(SNNLayer):
    r"""A dual synaptic leaky integrate-and-ternary-scaled-fire layer.

    ``DSLITS`` combines dual synaptic traces, dual membrane traces, and scaled
    ternary output. It is the scaled counterpart of ``DSLIT``.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
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
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
        learn_pos_alpha (bool, default=True): whether ``pos_alpha`` is trainable.
        learn_neg_alpha (bool, default=True): whether ``neg_alpha`` is trainable.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        pos_syn: positive synaptic state.
        neg_syn: negative synaptic state.
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Use this layer when sign-specific input smoothing and sign-specific
        output magnitudes are both useful, but no recurrent output trace is
        needed.

    Examples::

        >>> layer = tt.snn.DSLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("pos_syn", (num_features,))
        self.define_state("neg_syn", (num_features,))
        self.define_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self.define_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self.define_state("pos_mem", (num_features,))
        self.define_state("neg_mem", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.spike_fn = spike_fn

        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_syn = self.pos_syn
        neg_syn = self.neg_syn
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = pos_syn
        self.neg_syn = neg_syn

        pos_mem = pos_mem * self.pos_beta + pos_syn
        neg_mem = neg_mem * self.neg_beta + neg_syn

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        return scaled_spikes


class DRLITS(SNNLayer):
    r"""A dual recurrent leaky integrate-and-ternary-scaled-fire layer.

    ``DRLITS`` combines dual membrane traces, dual recurrent traces, and scaled
    ternary output. It is useful when positive and negative recurrent history
    should have different decays, gains, and downstream output magnitudes.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_gamma (float or torch.Tensor, default=0.9): positive recurrent decay.
        neg_gamma (float or torch.Tensor, default=0.9): negative recurrent decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
        pos_rec_weight (float or torch.Tensor, default=0.0): positive recurrent
            input scale.
        neg_rec_weight (float or torch.Tensor, default=0.0): negative recurrent
            input scale.
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
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
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
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
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
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        The two previous firing outputs are stored separately and unscaled,
        so reset and recurrence preserve both branches independently.

    Examples::

        >>> layer = tt.snn.DRLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            pos_rec_weight_rank: Literal[0, 1] = 1,
            neg_rec_weight_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
            learn_pos_gamma: bool = True,
            learn_neg_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("pos_mem", (num_features,))
        self.define_state("neg_mem", (num_features,))
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.define_state("pos_rec", (num_features,))
        self.define_state("neg_rec", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self.define_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn

        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self.define_unbound_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self.define_unbound_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_rec = self.pos_rec
        neg_rec = self.neg_rec

        pos_rec = pos_rec * self.pos_gamma + pos_prev_output * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + neg_prev_output * (1 - self.neg_gamma)

        self.pos_rec = pos_rec
        self.neg_rec = neg_rec

        pos_mem_delta = pos_rec * self.pos_rec_weight + torch.where(x >= 0, x, 0.0)
        neg_mem_delta = neg_rec * self.neg_rec_weight + torch.where(x <= 0, x, 0.0)

        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        return scaled_spikes


class SRLITS(SNNLayer):
    r"""A synaptic recurrent leaky integrate-and-ternary-scaled-fire layer.

    ``SRLITS`` combines a synaptic input trace, a recurrent output trace, and
    scaled ternary firing. It is the scaled counterpart of ``SRLIT``.

    Args:
        num_features (int): number of neurons in the final dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay.
        beta (float or torch.Tensor, default=0.9): membrane decay.
        gamma (float or torch.Tensor, default=0.9): recurrent trace decay.
        pos_threshold (float or torch.Tensor, default=1.0): positive threshold.
        neg_threshold (float or torch.Tensor, default=1.0): negative threshold
            magnitude.
            boundaries.
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
        rec_weight (float or torch.Tensor, default=0.0): recurrent input scale.
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
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
        rec_weight_rank (Literal[0, 1], default=1): scalar or per-neuron
            recurrent scale.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.
        learn_gamma (bool, default=True): whether ``gamma`` is trainable.
        learn_pos_threshold (bool, default=True): whether ``pos_threshold`` is
            trainable.
        learn_neg_threshold (bool, default=True): whether ``neg_threshold`` is
            trainable.
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
        learn_rec_weight (bool, default=True): whether ``rec_weight`` is trainable.
        spike_fn (Callable, default=tt.snn.spike_fn.deterministic): spike function.

    Attributes:
        syn: synaptic state.
        mem: membrane state.
        rec: recurrent trace state.
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            syn = alpha * syn + (1 - alpha) * x
            rec = gamma * rec + (1 - gamma) * (pos_prev_output + neg_prev_output)
            mem = beta * mem + syn + rec_weight * rec
            pos_prev_output = pos
            neg_prev_output = neg
            return pos * pos_scale + neg * neg_scale

    Examples::

        >>> layer = tt.snn.SRLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            rec_weight: Union[float, torch.Tensor] = 0.0,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
            rec_weight_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            learn_pos_threshold: bool = True,
            learn_neg_threshold: bool = True,
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("syn", (num_features,))
        self.define_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.define_state("mem", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

        self.define_state("rec", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("gamma", gamma, gamma_rank, learn_gamma)

        self.spike_fn = spike_fn

        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self.define_unbound_parameter("rec_weight", rec_weight, rec_weight_rank, learn_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        mem = self.mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        mem = mem - pos_prev_output * self.pos_threshold
        mem = mem - neg_prev_output * self.neg_threshold

        syn = self.syn
        syn = syn * self.alpha + x * (1 - self.alpha)

        rec = self.rec
        rec = rec * self.gamma + (pos_prev_output + neg_prev_output) * (1 - self.gamma)
        self.rec = rec

        mem_delta = rec * self.rec_weight + syn

        mem = mem * self.beta + mem_delta

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.syn = syn
        self.mem = mem

        return scaled_spikes


class DSRLITS(SNNLayer):
    r"""A dual synaptic recurrent leaky integrate-and-ternary-scaled-fire layer.

    ``DSRLITS`` combines every SNN mechanism provided by traceTorch: dual
    sign-specific synaptic traces, dual sign-specific recurrent traces, dual
    sign-specific membrane traces, ternary firing, and independent positive and
    negative output scales.

    Args:
        num_features (int): number of neurons in the final dimension.
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
        pos_scale (float or torch.Tensor, default=1.0): positive output scale.
        neg_scale (float or torch.Tensor, default=1.0): negative output scale.
        pos_rec_weight (float or torch.Tensor, default=0.0): positive recurrent
            input scale.
        neg_rec_weight (float or torch.Tensor, default=0.0): negative recurrent
            input scale.
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
        pos_scale_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            scale.
        neg_scale_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            scale.
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
        learn_pos_scale (bool, default=True): whether ``pos_scale`` is trainable.
        learn_neg_scale (bool, default=True): whether ``neg_scale`` is trainable.
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
        pos_prev_output: previous unscaled positive firing output.
        neg_prev_output: previous unscaled negative firing output.
        pos_scale: positive output scale.
        neg_scale: negative output scale.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        This layer is intentionally dense: it is meant for cases where sign,
        input history, output history, thresholding, and output magnitude all
        need separate degrees of freedom.

    Examples::

        >>> layer = tt.snn.DSRLITS(num_features=32)
        >>> input = torch.randn(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            pos_alpha: Union[float, torch.Tensor] = 0.5,
            neg_alpha: Union[float, torch.Tensor] = 0.5,
            pos_beta: Union[float, torch.Tensor] = 0.9,
            neg_beta: Union[float, torch.Tensor] = 0.9,
            pos_gamma: Union[float, torch.Tensor] = 0.9,
            neg_gamma: Union[float, torch.Tensor] = 0.9,
            pos_threshold: Union[float, torch.Tensor] = 1.0,
            neg_threshold: Union[float, torch.Tensor] = 1.0,
            pos_scale: Union[float, torch.Tensor] = 1.0,
            neg_scale: Union[float, torch.Tensor] = 1.0,
            pos_rec_weight: Union[float, torch.Tensor] = 0.0,
            neg_rec_weight: Union[float, torch.Tensor] = 0.0,
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            pos_gamma_rank: Literal[0, 1] = 1,
            neg_gamma_rank: Literal[0, 1] = 1,
            pos_threshold_rank: Literal[0, 1] = 1,
            neg_threshold_rank: Literal[0, 1] = 1,
            pos_scale_rank: Literal[0, 1] = 1,
            neg_scale_rank: Literal[0, 1] = 1,
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
            learn_pos_scale: bool = True,
            learn_neg_scale: bool = True,
            learn_pos_rec_weight: bool = True,
            learn_neg_rec_weight: bool = True,
            spike_fn=spike_functions.deterministic,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("pos_syn", (num_features,))
        self.define_state("neg_syn", (num_features,))
        self.define_decay("pos_alpha", pos_alpha, pos_alpha_rank, learn_pos_alpha)
        self.define_decay("neg_alpha", neg_alpha, neg_alpha_rank, learn_neg_alpha)

        self.define_state("pos_mem", (num_features,))
        self.define_state("neg_mem", (num_features,))
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

        self.define_state("pos_rec", (num_features,))
        self.define_state("neg_rec", (num_features,))
        self.define_state("pos_prev_output", (num_features,))
        self.define_state("neg_prev_output", (num_features,))
        self.define_decay("pos_gamma", pos_gamma, pos_gamma_rank, learn_pos_gamma)
        self.define_decay("neg_gamma", neg_gamma, neg_gamma_rank, learn_neg_gamma)

        self.spike_fn = spike_fn
        self.define_threshold("pos_threshold", pos_threshold, pos_threshold_rank, learn_pos_threshold)
        self.define_threshold("neg_threshold", neg_threshold, neg_threshold_rank, learn_neg_threshold)

        self.define_unbound_parameter("pos_scale", pos_scale, pos_scale_rank, learn_pos_scale)
        self.define_unbound_parameter("neg_scale", neg_scale, neg_scale_rank, learn_neg_scale)

        self.define_unbound_parameter("pos_rec_weight", pos_rec_weight, pos_rec_weight_rank, learn_pos_rec_weight)
        self.define_unbound_parameter("neg_rec_weight", neg_rec_weight, neg_rec_weight_rank, learn_neg_rec_weight)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_prev_output = self.pos_prev_output
        neg_prev_output = self.neg_prev_output
        pos_mem = pos_mem - pos_prev_output * self.pos_threshold * 0.5
        pos_mem = pos_mem - neg_prev_output * self.neg_threshold * 0.5
        neg_mem = neg_mem - pos_prev_output * self.pos_threshold * 0.5
        neg_mem = neg_mem - neg_prev_output * self.neg_threshold * 0.5

        pos_syn = self.pos_syn
        neg_syn = self.neg_syn
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = pos_syn
        self.neg_syn = neg_syn

        pos_rec = self.pos_rec
        neg_rec = self.neg_rec

        pos_rec = pos_rec * self.pos_gamma + pos_prev_output * (1 - self.pos_gamma)
        neg_rec = neg_rec * self.neg_gamma + neg_prev_output * (1 - self.neg_gamma)

        self.pos_rec = pos_rec
        self.neg_rec = neg_rec

        pos_mem_delta = pos_rec * self.pos_rec_weight + pos_syn
        neg_mem_delta = neg_rec * self.neg_rec_weight + neg_syn

        pos_mem = pos_mem * self.pos_beta + pos_mem_delta
        neg_mem = neg_mem * self.neg_beta + neg_mem_delta

        mem = pos_mem + neg_mem

        pos_spikes = self.spike_fn(mem - self.pos_threshold)
        neg_spikes = -self.spike_fn(-self.neg_threshold - mem)

        scaled_spikes = pos_spikes * self.pos_scale + neg_spikes * self.neg_scale

        self.pos_prev_output = pos_spikes
        self.neg_prev_output = neg_spikes
        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        return scaled_spikes
