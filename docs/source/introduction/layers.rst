Choosing from the built-in layers
=================================

Now that we've seen the state updates, we can connect them to the public layer
names. The SNN families share a feature-wise interface, so we can choose their
memory and firing behavior independently of the projections around them.
The conventional RNN layers include their own feature projections and gates.

SNN output families and histories
---------------------------------

The end of an SNN name describes its output. ``LI`` returns the continuous
membrane, ``LIB`` fires in one direction, ``LIT`` can fire in either direction,
and ``LITS`` also scales those signed outputs. The firing families use hard
deterministic events by default, but accept smooth or stochastic firing functions.

The prefixes describe additional histories: ``S`` filters input through a
synapse, ``R`` filters earlier output through a recurrent trace, and ``D``
separates positive and negative histories. For example, ``SRLIB`` combines
synaptic and recurrent histories with one-sided firing. The continuous family
also offers ``EMA`` variants for an averaged membrane update.

.. list-table::
   :header-rows: 1

   * - Family
     - Available layers
   * - Continuous membrane
     - ``LI``, ``DLI``, ``SLI``, ``DSLI``, ``LIEMA``, ``DLIEMA``, ``SLIEMA``, ``DSLIEMA``
   * - One-sided firing
     - ``LIB``, ``DLIB``, ``SLIB``, ``RLIB``, ``DSLIB``, ``DRLIB``, ``SRLIB``, ``DSRLIB``
   * - Signed firing
     - ``LIT``, ``DLIT``, ``SLIT``, ``RLIT``, ``DSLIT``, ``DRLIT``, ``SRLIT``, ``DSRLIT``
   * - Scaled signed firing
     - ``LITS``, ``DLITS``, ``SLITS``, ``RLITS``, ``DSLITS``, ``DRLITS``, ``SRLITS``, ``DSRLITS``

In the patch-MNIST tutorial, LIB hidden layers turn accumulated image information
into firing events, while the final LI retains continuous class scores. The
custom-neuron tutorial changes the synaptic histories while keeping the same
membrane and readout ideas, which shows how these choices can be composed.

Features and parameter constraints
----------------------------------

For built-in SNN layers, ``num_features`` is the size of the final input axis.
``tt.snn.LIB(64)`` therefore accepts ``[..., 64]`` and gives each leading position
its own states. A preceding ``nn.Linear`` can change the number of features or
mix them; the neuron then applies its dynamics to those projected values.

The decay names follow the history they control: ``alpha`` belongs to the
synapse, ``beta`` to the membrane, and ``gamma`` to recurrent output memory.
Their effective values stay in :math:`(0,1)` because the layer transforms
unconstrained raw parameters:

.. math::

   \beta=\sigma(b), \qquad
   b_{initial}=\log\frac{\beta_{initial}}{1-\beta_{initial}}.

The inverse transformation lets us request an effective initial value such as
0.9, while the optimizer works on :math:`b`. Thresholds and output scales use
softplus to stay positive:

.. math::

   \theta = \log(1+e^v).

We access these effective values through names such as ``layer.beta`` and
``layer.threshold``. Their raw tensors are registered PyTorch parameters, so
ordinary optimizers and weight checkpoints already include them.

A ``*_rank`` of 0 creates a shared scalar, while 1 creates one value per neuron.
We can also provide a scalar tensor or a vector with ``num_features`` elements
directly, in which case that tensor supplies the parameter shape. The generic
``tt.Layer`` base supports other shapes too; ``tt.snn.Layer(num_features)`` adds
the scalar/per-neuron conveniences used by built-in neurons.

The ``learn_*`` arguments let us hold a parameter fixed. For example,
``learn_beta=False`` keeps the selected membrane decay while other weights
train. If we change that choice later, ``set_parameter_learnable("beta", True)``
updates the intended setting, including across compilation and decompilation.

Conventional recurrent layers
-----------------------------

``tt.rnn.SimpleRNN``, ``GRU`` and ``LSTM`` have the same one-timestep state
interface, but learn projections that mix the input and previous hidden features
inside the recurrent layer. Their constructors take ``in_features`` and
``out_features``, and the output size also sets the hidden feature count.

For a SimpleRNN, concatenating the previous hidden vector :math:`h_{t-1}`
with the input gives the update

.. math::

   h_t=\tanh(W[h_{t-1},x_t]+b).

The brackets mean concatenation along the feature axis. Unlike a fixed LI decay,
this update mixes hidden features and applies a nonlinear function at every step.

A GRU adds reset and update gates, whose values depend on both the input and
the previous state. In traceTorch's convention:

.. math::

   [r_t,z_t] &= \sigma(W_g[h_{t-1},x_t]+b_g), \\
   \tilde h_t &= \tanh(W_c[r_t\odot h_{t-1},x_t]+b_c), \\
   h_t &= (1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.

Here :math:`\sigma` is sigmoid, and :math:`\odot` means element-wise
multiplication. The matrices :math:`W_g`, :math:`W_c` and biases :math:`b_g`,
:math:`b_c` are learned projections. The reset gate
:math:`r_t` selects how much old state enters the candidate, while :math:`z_t`
selects how much of that candidate replaces the old state. This gives the model
input-dependent control over memory. These gated layers use the same state
operations as the SNN layers, so they can be used with the same explicit sequence
loops even though their internal updates differ.

An LSTM keeps a cell state :math:`c_t` as well as its hidden output :math:`h_t`.
Its learned input, forget and output gates :math:`i_t`, :math:`f_t`, :math:`o_t`
control a candidate :math:`\tilde c_t` and the updates

.. math::

   c_t &= f_t\odot c_{t-1}+i_t\odot\tilde c_t, \\
   h_t &= o_t\odot\tanh(c_t).

Both cell and hidden state take part in the same model-level reset, detach and
save operations. We don't need a separate training interface for each kind of
recurrent memory.

Extending the library
---------------------

When the existing updates don't describe the states we need, ``tt.Layer`` lets
us declare those states and their tensor shapes directly. We can then define
transformed parameters and write one timestep of the update in ``forward``.
The :doc:`../tutorials/custom_layer` tutorial does this with a matrix of synaptic
histories, rather than assuming every state is a vector.

The remaining helpers support these updates: ``tt.inverse_fn`` supplies inverse
initialization transforms, ``tt.snn.spike_fn`` supplies firing functions, and
``tt.utils`` provides decay conversion and explicit axis movement. The
:doc:`../reference/index` gives their signatures when you need exact arguments.
