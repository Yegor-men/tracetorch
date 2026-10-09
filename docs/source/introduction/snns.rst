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

Binary firing layers subtract the previous output's threshold-scaled reset before decay and integration:

::

    mem = (mem - prev_output * threshold) * beta + x
    spikes = spike_fn(mem - threshold)
    prev_output = spikes
    return spikes

The default ``spike_fn`` is ``tt.snn.spike_fn.deterministic``: a hard threshold in the forward
pass with the derivative of ``sigmoid(4 * x)`` in the backward pass. Use
``tt.snn.spike_fn.smooth`` for continuous firing intensities or ``tt.snn.spike_fn.stochastic``
for stochastic events.

Ternary layers retain two signed, unscaled event states, ``pos_prev_output`` and
``neg_prev_output``, rather than storing their sum. Their reset uses both events:

::

    reset = pos_prev_output * pos_threshold + neg_prev_output * neg_threshold
    mem = (mem - reset) * beta + x
    pos_spikes = spike_fn(mem - pos_threshold)
    neg_spikes = -spike_fn(-neg_threshold - mem)
    pos_prev_output = pos_spikes
    neg_prev_output = neg_spikes
    return pos_spikes + neg_spikes

This preserves both firing intensities when ``spike_fn`` is smooth, and both
events if stochastic firing activates both branches. ``LITS`` applies
``pos_scale`` and ``neg_scale`` only to the returned output; reset and recurrence
always use the unscaled events. Dual membrane layers share the reset equally
between their two membrane traces.

Surrogate gradients
-------------------

Hard spikes are difficult to train with ordinary gradient descent because a step function has zero derivative almost
everywhere and is undefined at the threshold. Surrogate-gradient training keeps the forward computation spike-like while
using a smoother backward approximation.

In traceTorch, the separation is explicit:

* ``spike_fn`` turns membrane distance from threshold into the returned firing output.
* ``smooth`` returns a continuous firing intensity.
* ``deterministic`` and ``stochastic`` return hard spikes with a smooth surrogate derivative.

This design makes the training behavior visible. The default keeps the forward pass discrete and uses a continuous backward approximation. The hard
spike functions make the forward pass discrete while keeping the backward pass trainable.

Synaptic traces
---------------

Synaptic variants add a trace called ``syn`` before the membrane:

::

    syn = alpha * syn + (1 - alpha) * x
    mem = beta * mem + syn

This makes the input current smoother before it reaches the membrane. Dual synaptic layers use ``pos_syn`` and
``neg_syn`` so positive and negative signals can decay independently. Each
synaptic trace feeds its corresponding membrane trace directly, without summing
the two currents and splitting that sum by sign again.

Recurrent traces
----------------

Recurrent variants add a trace of the previous output:

::

    rec = gamma * rec + (1 - gamma) * prev_output
    mem = beta * mem + x + rec_weight * rec

The recurrent trace is internal to the layer. The caller still passes one tensor in and receives one tensor out, which
keeps traceTorch layers composable with ordinary PyTorch modules.

For ternary layers, a single ``rec`` trace filters the sum of the two unscaled
events. Dual recurrent layers instead filter each event independently:

::

    pos_rec = pos_gamma * pos_rec + (1 - pos_gamma) * pos_prev_output
    neg_rec = neg_gamma * neg_rec + (1 - neg_gamma) * neg_prev_output

Each recurrent trace feeds its corresponding membrane through its own recurrent
weight, without combining and re-splitting the histories.

Ternary hidden-state checkpoints use ``pos_prev_output`` and ``neg_prev_output``;
older checkpoints containing only ``prev_output`` do not preserve both branches
and cannot reconstruct them in general. Parameter names in weight checkpoints
are unchanged.

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

This means that ``SRLIT`` is a synaptic recurrent ternary layer, while ``DLITS`` is a dual ternary-scaled layer.

Common parameters
-----------------

Most SNN parameters follow the same rules across all layers.

``num_features`` is the final input dimension size. For example, ``tt.snn.LIB(64)`` accepts ``[..., 64]``. Move channel-first images explicitly before applying the layer.

Decay parameters are constrained to ``(0, 1)``:

* ``alpha`` controls synaptic memory.
* ``beta`` controls membrane memory.
* ``gamma`` controls recurrent-output memory.

Threshold parameters are constrained to positive values:

* ``threshold`` is used by binary layers.
* ``pos_threshold`` and ``neg_threshold`` are used by ternary layers.

The ``*_rank`` arguments choose whether a parameter is shared or per-neuron. A rank of ``0`` creates a scalar. A rank of
``1`` creates a vector of length ``num_features``. You can also pass a tensor directly; scalar tensors are accepted, and
1D tensors must have ``num_features`` elements.

The ``learn_*`` arguments choose whether a parameter is trainable. If ``learn_beta=False``, for example, the raw decay is
stored as an ``nn.Parameter`` with ``requires_grad=False``. Use ``set_parameter_learnable`` to change intended learnability later, including across compilation.

Working on non-last dimensions
------------------------------

Built-in layers operate on the final dimension. For channel-first images, move the channel axis explicitly:

::

    layer = nn.Sequential(
        tt.utils.MoveDim(-3, -1),
        tt.snn.LIB(num_features=32),
        tt.utils.MoveDim(-1, -3),
    )
    x = torch.rand(16, 32, 28, 28)
    y = layer(x)
    # y.shape == torch.Size([16, 32, 28, 28])

States remain in feature-last layout; only the layer's input and output are moved.

Practical choices
-----------------

Start simple unless the task gives you a reason not to.

Use ``LI`` as a readout or continuous accumulator when you want to keep magnitude information. Use ``LIEMA`` when the
trace should remain bounded like a smoothed signal.

Use ``LIB`` for ordinary one-sided SNN experiments. It is the traceTorch name for the common leaky integrate-and-fire
shape, with hard spikes and surrogate gradients by default.

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
