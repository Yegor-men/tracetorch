from typing import TypedDict, Optional, Literal, Union, Dict, Any
import torch
from torch import nn
from ._snnlayer import Layer as SNNLayer


class LI(SNNLayer):
    r"""A leaky integrator layer with continuous membrane output.

    ``LI`` stores a membrane trace and returns that trace directly. It does not
    spike, threshold, or reset; it is useful as a readout-style accumulator or as
    a smooth recurrent feature transform.

    Args:
        num_features (int): number of neurons in the final dimension.
        beta (float or torch.Tensor, default=0.9): membrane decay. The activated
            value is constrained to ``(0, 1)``.
        beta_rank (Literal[0, 1], default=1): ``0`` for a scalar decay shared by
            all neurons, ``1`` for one decay per neuron.
        learn_beta (bool, default=True): whether ``beta`` is trainable.

    Attributes:
        mem: membrane state. Lazily initialized to zeros with the input shape.
        beta: activated membrane decay.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Updates the membrane by exponentially decaying the previous value and
        adding the current input. Pseudocode looks as follows:

        ::

            mem = beta * mem + x
            return mem

    Examples::

        >>> layer = tt.snn.LI(num_features=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            beta: Union[float, torch.Tensor] = 0.9,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("mem", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        mem = self.mem
        mem = mem * self.beta + x

        self.mem = mem

        return self.mem


class DLI(SNNLayer):
    r"""A dual leaky integrator layer with continuous membrane output.

    ``DLI`` splits the membrane trace into separate positive and negative
    branches. Positive input updates ``pos_mem`` and negative input updates
    ``neg_mem``; the returned membrane is their sum.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_beta (float or torch.Tensor, default=0.9): decay for the positive
            membrane branch.
        neg_beta (float or torch.Tensor, default=0.9): decay for the negative
            membrane branch.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            decay.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.

    Attributes:
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.
        pos_beta: activated positive membrane decay, constrained to ``(0, 1)``.
        neg_beta: activated negative membrane decay, constrained to ``(0, 1)``.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Dual traces let positive and negative evidence retain different time
        constants. Pseudocode looks as follows:

        ::

            pos_mem = pos_beta * pos_mem + where(x >= 0, x, 0)
            neg_mem = neg_beta * neg_mem + where(x <= 0, x, 0)
            return pos_mem + neg_mem

    Examples::

        >>> layer = tt.snn.DLI(num_features=32)
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
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("pos_mem", (num_features,))
        self.define_state("neg_mem", (num_features,))
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_mem = pos_mem * self.pos_beta + torch.where(x >= 0, x, 0.0)
        neg_mem = neg_mem * self.neg_beta + torch.where(x <= 0, x, 0.0)

        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        mem = self.pos_mem + self.neg_mem

        return mem


class SLI(SNNLayer):
    r"""A synaptic leaky integrator layer with continuous membrane output.

    ``SLI`` adds a synaptic trace before the membrane trace. The synaptic trace
    smooths the input with decay ``alpha`` before the membrane integrates it
    with decay ``beta``.

    Args:
        num_features (int): number of neurons in the final dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay, constrained
            to ``(0, 1)``.
        beta (float or torch.Tensor, default=0.9): membrane decay, constrained
            to ``(0, 1)``.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.

    Attributes:
        syn: synaptic state.
        mem: membrane state.
        alpha: activated synaptic decay.
        beta: activated membrane decay.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        The synaptic trace is an exponential moving average of the input. The
        membrane then accumulates that smoothed current. Pseudocode looks as
        follows:

        ::

            syn = alpha * syn + (1 - alpha) * x
            mem = beta * mem + syn
            return mem

    Examples::

        >>> layer = tt.snn.SLI(num_features=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("syn", (num_features,))
        self.define_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.define_state("mem", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        syn = self.syn
        syn = syn * self.alpha + x * (1 - self.alpha)

        mem = self.mem
        mem = mem * self.beta + syn

        self.syn = syn
        self.mem = mem

        return self.mem


class DSLI(SNNLayer):
    r"""A dual synaptic leaky integrator layer with continuous membrane output.

    ``DSLI`` combines dual positive/negative traces with a synaptic stage. It
    keeps separate positive and negative synaptic traces, then integrates each
    directly into its corresponding membrane trace.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            synaptic decay.
        neg_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            synaptic decay.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            membrane decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            membrane decay.
        learn_pos_alpha (bool, default=True): whether ``pos_alpha`` is trainable.
        learn_neg_alpha (bool, default=True): whether ``neg_alpha`` is trainable.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.

    Attributes:
        pos_syn: positive synaptic state.
        neg_syn: negative synaptic state.
        pos_mem: positive membrane state.
        neg_mem: negative membrane state.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            pos_syn = pos_alpha * pos_syn + (1 - pos_alpha) * where(x >= 0, x, 0)
            neg_syn = neg_alpha * neg_syn + (1 - neg_alpha) * where(x <= 0, x, 0)
            pos_mem = pos_beta * pos_mem + pos_syn
            neg_mem = neg_beta * neg_mem + neg_syn
            return pos_mem + neg_mem

    Examples::

        >>> layer = tt.snn.DSLI(num_features=32)
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
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
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

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_syn = self.pos_syn
        neg_syn = self.neg_syn
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = pos_syn
        self.neg_syn = neg_syn

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_mem = pos_mem * self.pos_beta + pos_syn
        neg_mem = neg_mem * self.neg_beta + neg_syn

        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        mem = self.pos_mem + self.neg_mem

        return mem


class LIEMA(SNNLayer):
    r"""A leaky integrator layer with exponential-moving-average output.

    ``LIEMA`` is the bounded counterpart to ``LI``. It stores one membrane trace
    and updates it as an exponential moving average of the input instead of as an
    unnormalized accumulator.

    Args:
        num_features (int): number of neurons in the final dimension.
        beta (float or torch.Tensor, default=0.9): membrane EMA decay,
            constrained to ``(0, 1)``.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron decay.
        learn_beta (bool, default=True): whether ``beta`` is trainable.

    Attributes:
        mem: membrane EMA state.
        beta: activated membrane decay.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            mem = beta * mem + (1 - beta) * x
            return mem

    Examples::

        >>> layer = tt.snn.LIEMA(num_features=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            beta: Union[float, torch.Tensor] = 0.9,
            beta_rank: Literal[0, 1] = 1,
            learn_beta: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("mem", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        mem = self.mem
        mem = mem * self.beta + x * (1 - self.beta)

        self.mem = mem

        return self.mem


class DLIEMA(SNNLayer):
    r"""A dual leaky integrator layer with exponential-moving-average output.

    ``DLIEMA`` keeps positive and negative membrane EMA traces separately and
    returns their sum. This is useful when positive and negative evidence should
    decay independently without allowing unbounded accumulation.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane EMA
            decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane EMA
            decay.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            decay.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.

    Attributes:
        pos_mem: positive membrane EMA state.
        neg_mem: negative membrane EMA state.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            pos_mem = pos_beta * pos_mem + (1 - pos_beta) * where(x >= 0, x, 0)
            neg_mem = neg_beta * neg_mem + (1 - neg_beta) * where(x <= 0, x, 0)
            return pos_mem + neg_mem

    Examples::

        >>> layer = tt.snn.DLIEMA(num_features=32)
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
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("pos_mem", (num_features,))
        self.define_state("neg_mem", (num_features,))
        self.define_decay("pos_beta", pos_beta, pos_beta_rank, learn_pos_beta)
        self.define_decay("neg_beta", neg_beta, neg_beta_rank, learn_neg_beta)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_mem = pos_mem * self.pos_beta + torch.where(x >= 0, x, 0.0) * (1 - self.pos_beta)
        neg_mem = neg_mem * self.neg_beta + torch.where(x <= 0, x, 0.0) * (1 - self.neg_beta)

        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        mem = self.pos_mem + self.neg_mem

        return mem


class SLIEMA(SNNLayer):
    r"""A synaptic leaky integrator layer with exponential-moving-average output.

    ``SLIEMA`` smooths the input through a synaptic EMA and then updates the
    membrane as an EMA of that synaptic current.

    Args:
        num_features (int): number of neurons in the final dimension.
        alpha (float or torch.Tensor, default=0.5): synaptic decay, constrained
            to ``(0, 1)``.
        beta (float or torch.Tensor, default=0.9): membrane decay, constrained
            to ``(0, 1)``.
        alpha_rank (Literal[0, 1], default=1): scalar or per-neuron synaptic
            decay.
        beta_rank (Literal[0, 1], default=1): scalar or per-neuron membrane
            decay.
        learn_alpha (bool, default=True): whether ``alpha`` is trainable.
        learn_beta (bool, default=True): whether ``beta`` is trainable.

    Attributes:
        syn: synaptic EMA state.
        mem: membrane EMA state.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            syn = alpha * syn + (1 - alpha) * x
            mem = beta * mem + (1 - beta) * syn
            return mem

    Examples::

        >>> layer = tt.snn.SLIEMA(num_features=32)
        >>> input = torch.rand(16, 32)
        >>> output = layer(input)
        >>> print(output.shape)
        torch.Size([16, 32])
    """

    def __init__(
            self,
            num_features: int,
            alpha: Union[float, torch.Tensor] = 0.5,
            beta: Union[float, torch.Tensor] = 0.9,
            alpha_rank: Literal[0, 1] = 1,
            beta_rank: Literal[0, 1] = 1,
            learn_alpha: bool = True,
            learn_beta: bool = True,
    ):
        super().__init__()
        self.num_features = num_features

        self.define_state("syn", (num_features,))
        self.define_decay("alpha", alpha, alpha_rank, learn_alpha)

        self.define_state("mem", (num_features,))
        self.define_decay("beta", beta, beta_rank, learn_beta)

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        syn = self.syn
        syn = syn * self.alpha + x * (1 - self.alpha)

        mem = self.mem
        mem = mem * self.beta + syn * (1 - self.beta)

        self.syn = syn
        self.mem = mem

        return self.mem


class DSLIEMA(SNNLayer):
    r"""A dual synaptic leaky integrator layer with EMA membrane output.

    ``DSLIEMA`` is the dual, synaptic, bounded-output variant of the leaky
    integrator family. It keeps positive and negative synaptic traces, then
    updates each membrane EMA trace directly from its corresponding synaptic
    current.

    Args:
        num_features (int): number of neurons in the final dimension.
        pos_alpha (float or torch.Tensor, default=0.5): positive synaptic decay.
        neg_alpha (float or torch.Tensor, default=0.5): negative synaptic decay.
        pos_beta (float or torch.Tensor, default=0.9): positive membrane decay.
        neg_beta (float or torch.Tensor, default=0.9): negative membrane decay.
        pos_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            synaptic decay.
        neg_alpha_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            synaptic decay.
        pos_beta_rank (Literal[0, 1], default=1): scalar or per-neuron positive
            membrane decay.
        neg_beta_rank (Literal[0, 1], default=1): scalar or per-neuron negative
            membrane decay.
        learn_pos_alpha (bool, default=True): whether ``pos_alpha`` is trainable.
        learn_neg_alpha (bool, default=True): whether ``neg_alpha`` is trainable.
        learn_pos_beta (bool, default=True): whether ``pos_beta`` is trainable.
        learn_neg_beta (bool, default=True): whether ``neg_beta`` is trainable.

    Attributes:
        pos_syn: positive synaptic EMA state.
        neg_syn: negative synaptic EMA state.
        pos_mem: positive membrane EMA state.
        neg_mem: negative membrane EMA state.

    Notes:
        - **Input**: tensor of shape ``[..., num_features]``.
        - **Output**: tensor with the same shape as the input.

        Pseudocode looks as follows:

        ::

            pos_syn = pos_alpha * pos_syn + (1 - pos_alpha) * where(x >= 0, x, 0)
            neg_syn = neg_alpha * neg_syn + (1 - neg_alpha) * where(x <= 0, x, 0)
            pos_mem = pos_beta * pos_mem + (1 - pos_beta) * pos_syn
            neg_mem = neg_beta * neg_mem + (1 - neg_beta) * neg_syn
            return pos_mem + neg_mem

    Examples::

        >>> layer = tt.snn.DSLIEMA(num_features=32)
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
            pos_alpha_rank: Literal[0, 1] = 1,
            neg_alpha_rank: Literal[0, 1] = 1,
            pos_beta_rank: Literal[0, 1] = 1,
            neg_beta_rank: Literal[0, 1] = 1,
            learn_pos_alpha: bool = True,
            learn_neg_alpha: bool = True,
            learn_pos_beta: bool = True,
            learn_neg_beta: bool = True,
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

    def forward(self, x):
        """Computes the forward pass."""
        self.zero_states(x)

        pos_syn = self.pos_syn
        neg_syn = self.neg_syn
        pos_syn = pos_syn * self.pos_alpha + torch.where(x >= 0, x, 0.0) * (1 - self.pos_alpha)
        neg_syn = neg_syn * self.neg_alpha + torch.where(x <= 0, x, 0.0) * (1 - self.neg_alpha)

        self.pos_syn = pos_syn
        self.neg_syn = neg_syn

        pos_mem = self.pos_mem
        neg_mem = self.neg_mem
        pos_mem = pos_mem * self.pos_beta + pos_syn * (1 - self.pos_beta)
        neg_mem = neg_mem * self.neg_beta + neg_syn * (1 - self.neg_beta)

        self.pos_mem = pos_mem
        self.neg_mem = neg_mem

        mem = self.pos_mem + self.neg_mem

        return mem
