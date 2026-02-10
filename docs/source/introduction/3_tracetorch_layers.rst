3. The traceTorch Layers
========================

It is time to (briefly) take a look at all the layer types that traceTorch presents. The practical application of the layers
and when the right one should be used is discussed in more detail in the :doc:`tutorials <../tutorials/index>` section,
here we look at the toggleable features available.

As mentioned before, all traceTorch layers are small wrappers of the ``LeakyIntegrator``, which handles the optional features
via toggleable flags. All layers sit in ``tracetorch.snn``. Detailed explanations of each layer can be found in the
:doc:`references <../reference/index>` section.

The simplest of layers is the ``LIF`` - Leaky Integrate and Fire. Internally, it holds the following states/parameters:
 - ``mem``: the membrane potential which stores the charge
 - ``beta``: the decay factor applied to ``mem`` at each timestep, bound to (0,1) via sigmoid
 - ``pos_threshold``: the threshold that ``mem`` needs to surpass in order to produce a positive spike, bound to (0,inf) via softplus

At each timestep, ``mem`` is updated as such: ``mem = mem * beta + mem_delta``, where ``mem_delta`` is the output of the
upstream layer (most likely a PyTorch layer). If a neuron fires by the ``mem`` surpassing the threshold, the threshold
amount is subtracted from ``mem`` at that neuron to reset it. As the threshold is always bound to be positive, this makes
for easy interpretability of signals: positive is excitatory and negative is inhibitory.

The second simplest layer is ``Readout``. It is identical to the ``LIF`` layer, but without any thresholds. Since there's
no threshold to reset ``mem``, we now make ``mem`` work as an Exponential Moving Average (EMA) by scaling ``mem_delta``
by ``1 - beta``. Had we not done this, the magnitude that ``mem`` reaches is directly affected by the ``beta`` decay. For example,
if ``beta`` was 0.9 and the average ``mem_delta`` was 1.0, then the value we'd expect ``mem`` to stabilize at would be
``mem_delta / (1 - beta)`` which is 10. If ``beta`` were 0.99, then it would stabilize at 100, and so on. ``beta`` should
modulate the timescale that the layer looks at, not the magnitude stored, thus we make ``mem`` an EMA so that regardless
of ``beta``, the value is a true reflection of the moving average. The output is ``mem`` itself, a float, no spikes.
Thus, at each timestep, ``mem`` is updated as such: ``mem = mem * beta + mem_delta * (1 - beta)``.

It's important to note that ``Readout`` is *not* recommended to be the final layer in the model. Despite the naming (just a convention),
a far better practice is to have another PyTorch layer right afterwards (such as ``nn.Linear``), to create a weighted average
of the timescales the model is interested in, not a literal rolling average of the model's outputs.

``LIF`` and ``Readout`` are the simplest of layer types. They are stateful, and certainly work for simpler tasks, but are
not the peak of possible power. tracetorch layer names are modular, various letters meaning various features, absence of
a letter means an absence of the feature:
 - ``S``: An extra synapse layer ``syn`` before ``mem``. Inputs are accumulated there first, and ``syn`` decays with
   ``alpha``. It is then ``syn`` that integrates the current into ``mem`` (``mem_delta`` is the current ``syn``). Since
   ``syn`` has no reset mechanism, it's an EMA for numerical stability, as otherwise ``mem_delta`` would make ``mem``
   overflow way beyond the threshold.
 - ``R``: An extra recurrence layer, where the outputs from each timestep are recorded into ``rec``, an EMA trace akin
   to ``syn`` that decays with ``gamma``. At each timestep, to ``mem_delta`` is added is the current ``rec``. There is
   also ``bias``, a value that's just added to ``mem_delta`` at each timestep.
 - ``B``: Bipolar spiking, outputs are balanced ternary (-1/0/1) instead of binary (0/1). There's now a ``pos_threshold``
   and a ``neg_threshold`` bound between (0,inf) and (-inf,0). If ``mem`` surpasses either threshold in the respective
   magnitude, it sends out a spike of the respective polarity, and resets by adding or subtracting the respective amount
   to get ``mem`` closer to 0. There is also ``pos_scale`` and ``neg_scale`` which scale the positive and negative spikes,
   so the output technically isn't balanced ternary. This is done so that the 3 outputs are truly independent of one another
   (this is not enabled for non-bipolar layers, as there it's not a problem).

These combinations result in the following 12 default layers:
 #. ``snn.Readout`` - Readout
 #. ``snn.SReadout`` - Synaptic Readout
 #. ``snn.RReadout`` - Recurrent Readout
 #. ``snn.SRReadout`` - Synaptic Recurrent Readout
 #. ``snn.LIF`` - LIF
 #. ``snn.SLIF`` - Synaptic LIF
 #. ``snn.RLIF`` - Recurrent LIF
 #. ``snn.SRLIF`` - Synaptic recurrent LIF
 #. ``snn.BLIF`` - Bipolar LIF
 #. ``snn.BSLIF`` - Bipolar Synaptic LIF
 #. ``snn.BRLIF`` - Bipolar Recurrent LIF
 #. ``snn.BSRLIF`` - Bipolar Synaptic Recurrent LIF

When initializing any layer, you can also specify the ``dim`` dimension. It's an index as to what dimension the layer should
focus on, which defaults to ``-1``, easily working with MLP networks. Change it to ``-3`` and the layer now works on the color
channel of ``[C, H, W]`` or ``[B, C, H, W]`` tensors. Change it to ``-4`` for example, and the layer works with colored voxel tensors: ``[C, X, Y, Z]``.
Each layer also asks for ``num_neurons``, this value should be the number of values at the desired ``dim`` dimension.
So in the ``dim=-3`` example, it would be the number of color channels.

When initializing any layer layer, each parameter also optionally asks for the following arguments:
 - ``value``: the actual value it should take (instead of ``value``, it's just the name of the parameter, such as ``beta``
   or ``pos_threshold``, et cetera.). Can be a ``float`` for automatic initialization or ``torch.Tensor`` for a custom tensor.
 - ``*_rank``: the rank of the parameter. 0 means scalar (per-layer), 1 means vector (per-neuron). Defaults to 1 (vector).
 - ``*_learnable``: should the parameter be learnable or static. Defaults to ``True``.
