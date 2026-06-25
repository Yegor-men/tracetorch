The SNNs of traceTorch
======================

traceTorch was designed around spiking neural networks, but it deliberately treats "spiking" as a configurable part of
the layer rather than as an unavoidable hard-coded event. The SNN layers are recurrent dynamical systems first: they keep
one or more traces, update those traces one timestep at a time, and optionally convert the membrane value into binary,
ternary, or scaled ternary output.

This is the main idea to keep in mind when reading the API:

* ``LI`` layers return the membrane trace directly. They are leaky integrators, not firing neurons.
* ``LIB`` layers use one positive threshold and produce non-negative output.
* ``LIT`` layers use positive and negative thresholds and can produce positive or negative output.
* ``LITS`` layers are ternary layers with learnable or fixed positive and negative output scales.
* ``D`` splits traces into positive and negative branches.
* ``S`` adds a synaptic trace before the membrane trace.
* ``R`` adds a recurrent trace from the previous output.

The membrane trace
------------------

The simplest traceTorch SNN layer is ``LI``. It stores a membrane state called ``mem`` and updates it as:

::

    mem = beta * mem + x
    return mem

The decay ``beta`` is constrained to ``(0, 1)`` by the layer registration machinery, even when it is learnable. A value
near zero makes the layer mostly react to the current input. A value near one gives the layer longer memory.

The ``LIEMA`` variants use an exponential-moving-average form instead:

::

    mem = beta * mem + (1 - beta) * x
    return mem

This keeps the magnitude bounded in a way that is useful when the layer is meant to smooth a signal instead of
accumulate evidence.

Spikes, intensities, and surrogate functions
--------------------------------------------

The firing layers first update a membrane trace and then compare it with one or more thresholds. For a binary ``LIB``
layer, the default calculation is:

::

    mem = beta * mem + x
    spikes = sigmoid4x(mem - threshold + bias)
    mem = mem - spikes * threshold
    return spikes

The default ``spike_fn`` is ``sigmoid4x``:

::

    sigmoid4x(x) = sigmoid(4 * x)

This gives a smooth transition around threshold. It is steep enough to behave like a soft firing decision, but it still
has useful gradients for optimization.

The default layer returns this smooth firing intensity itself, not a hard ``0`` or ``1`` event. In this default mode,
the SNN behaves more like a differentiable recurrent ODE-style system than a strictly discrete spiking mechanism. This is
intentional: it gives stable gradients and makes the default layer easy to train.

To make the forward pass more spike-like, pass a hard spike function explicitly:

::

    layer = tt.snn.LIB(
        num_neurons=128,
        spike_fn=tt.functional.round_sigmoid4x,
    )

traceTorch provides hard spike functions in ``tt.functional``. These functions make a discrete or stochastic forward
decision while defining a ``sigmoid4x`` surrogate gradient for the backward pass.

Surrogate gradients
-------------------

Hard spikes are difficult to train with ordinary gradient descent because a step function has zero derivative almost
everywhere and is undefined at the threshold. Surrogate-gradient training keeps the forward computation spike-like while
using a smoother backward approximation.

In traceTorch, the separation is explicit:

* ``spike_fn`` turns membrane distance from threshold into the returned firing output.
* ``sigmoid4x`` returns a continuous firing intensity.
* ``round_sigmoid4x`` and ``stochastic_sigmoid4x`` return hard spikes with a ``sigmoid4x`` surrogate derivative.

This design makes the training behavior visible. The default keeps the forward and backward paths continuous. The hard
spike functions make the forward pass discrete while keeping the backward pass trainable.

Synaptic traces
---------------

Synaptic variants add a trace called ``syn`` before the membrane:

::

    syn = alpha * syn + (1 - alpha) * x
    mem = beta * mem + syn

This makes the input current smoother before it reaches the membrane. Dual synaptic layers use ``pos_syn`` and
``neg_syn`` so positive and negative signals can decay independently.

Recurrent traces
----------------

Recurrent variants add a trace of the previous output:

::

    rec = gamma * rec + (1 - gamma) * prev_output
    mem = beta * mem + x + rec_weight * rec

The recurrent trace is internal to the layer. The caller still passes one tensor in and receives one tensor out, which
keeps traceTorch layers composable with ordinary PyTorch modules.

Choosing a layer
----------------

Use ``LI`` or ``LIEMA`` when you want a continuous trace. Use ``LIB`` when you want one-sided firing. Use ``LIT`` when
positive and negative events should be represented separately. Use ``LITS`` when the positive and negative events should
also have their own output magnitudes.

Then add prefixes only when they match the dynamics you need:

* Add ``D`` when positive and negative history should be stored separately.
* Add ``S`` when inputs should be smoothed before membrane integration.
* Add ``R`` when the neuron's previous output should influence its next membrane update.

The names look dense at first, but they are meant to be mechanical. ``DSRLITS`` is a dual, synaptic, recurrent, leaky
integrator with ternary scaled output.

The layer families
------------------

The 32 SNN layers are easier to understand as four output families, each with the same optional dynamics layered on top.

.. list-table::
   :header-rows: 1

   * - Family
     - Output
     - Base layer
     - Variants
   * - ``LI``
     - Continuous membrane value
     - ``LI``
     - ``LI``, ``DLI``, ``SLI``, ``DSLI``, ``LIEMA``, ``DLIEMA``, ``SLIEMA``, ``DSLIEMA``
   * - ``LIB``
     - One-sided non-negative firing value
     - ``LIB``
     - ``LIB``, ``DLIB``, ``SLIB``, ``RLIB``, ``DSLIB``, ``DRLIB``, ``SRLIB``, ``DSRLIB``
   * - ``LIT``
     - Positive, zero, or negative firing value
     - ``LIT``
     - ``LIT``, ``DLIT``, ``SLIT``, ``RLIT``, ``DSLIT``, ``DRLIT``, ``SRLIT``, ``DSRLIT``
   * - ``LITS``
     - Positive, zero, or negative firing value with separate output scales
     - ``LITS``
     - ``LITS``, ``DLITS``, ``SLITS``, ``RLITS``, ``DSLITS``, ``DRLITS``, ``SRLITS``, ``DSRLITS``

The output family goes at the end of the name. Prefixes describe extra internal traces:

* No prefix: one membrane trace.
* ``D``: positive and negative traces are stored separately.
* ``S``: a synaptic trace is added before the membrane.
* ``R``: a recurrent trace of the previous output is added before the membrane.
* ``DS``, ``DR``, ``SR``, ``DSR``: combinations of the above.

This means that ``SRLIT`` is a synaptic recurrent ternary layer, while ``DLITS`` is a dual ternary-scaled layer with
synaptic and recurrent traces.

Common parameters
-----------------

Most SNN parameters follow the same rules across all layers.

``num_neurons`` is the size of the dimension the layer operates on. The layer expects the input to already have this
size at ``dim``. For example, ``tt.snn.LIB(64, dim=-3)`` works on the channel dimension of a ``[B, C, H, W]`` tensor
where ``C == 64``.

Decay parameters are constrained to ``(0, 1)``:

* ``alpha`` controls synaptic memory.
* ``beta`` controls membrane memory.
* ``gamma`` controls recurrent-output memory.

Threshold parameters are constrained to positive values:

* ``threshold`` is used by binary layers.
* ``pos_threshold`` and ``neg_threshold`` are used by ternary layers.

The ``*_rank`` arguments choose whether a parameter is shared or per-neuron. A rank of ``0`` creates a scalar. A rank of
``1`` creates a vector of length ``num_neurons``. You can also pass a tensor directly; scalar tensors are accepted, and
1D tensors must have ``num_neurons`` elements.

The ``learn_*`` arguments choose whether a parameter is trainable. If ``learn_beta=False``, for example, the raw decay is
registered as a buffer rather than as an ``nn.Parameter``.

Working on non-last dimensions
------------------------------

traceTorch layers do not require features to live in the last dimension. The layer temporarily moves ``dim`` to the end,
runs the same vectorized update, then moves it back.

::

    layer = tt.snn.LIB(num_neurons=32, dim=-3)
    x = torch.rand(16, 32, 28, 28)
    y = layer(x)
    print(y.shape)
    # torch.Size([16, 32, 28, 28])

This is why the same SNN layer can be placed after a convolution, inside an MLP, or anywhere else that has a clear
feature dimension.

Practical choices
-----------------

Start simple unless the task gives you a reason not to.

Use ``LI`` as a readout or continuous accumulator when you want to keep magnitude information. Use ``LIEMA`` when the
trace should remain bounded like a smoothed signal.

Use ``LIB`` for ordinary one-sided SNN experiments. It is the traceTorch name for the common leaky integrate-and-fire
shape, with the important caveat that the default output is continuous unless you pass a quantizer.

Use ``LIT`` when negative events are meaningful. This is often more natural for signed activations, residual streams, or
signals where "below baseline" should be represented explicitly rather than merely suppressing positive firing.

Use ``LITS`` when the positive and negative events should have independent magnitudes. This lets the layer learn or fix
the downstream strength of positive and negative events separately from the thresholds that produced them.

Add ``S`` when input should arrive as a current over time instead of as a direct membrane increment. Add ``R`` when a
neuron's previous output should influence its next update. Add ``D`` when positive and negative history should have
separate time constants.

For a first model, something like this is usually easier to reason about than starting with the largest layer:

::

    self.net = nn.Sequential(
        nn.Linear(784, 128),
        tt.snn.LIB(128),
        nn.Linear(128, 10),
        tt.snn.LI(10),
    )

Once that works, the layer name gives a mechanical upgrade path: ``LIB`` to ``SLIB`` for input smoothing, ``LIB`` to
``RLIB`` for recurrent output memory, ``LIB`` to ``LIT`` for signed events, or ``LIT`` to ``LITS`` for signed events with
separate output magnitudes.
