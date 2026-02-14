3. The traceTorch Layers
========================

It is time to (briefly) take a look at the base layer types that traceTorch presents. The practical application of the layers
and when the right one should be used is discussed in more detail in the :doc:`tutorials <../tutorials/index>` section,
here we look at the toggleable features available.

traceTorch follows a slightly unconventional, but incredibly consistent and self-explanatory naming schema, encapsulating a vast
array of various neuron mechanics. All layers sit in ``tracetorch.snn``. Detailed explanations of each layer can be found in the
:doc:`references <../reference/index>` section.

- ``LI`` base name stands for ``Leaky Integrator``, the simplest of layer types with just one trace: the membrane potential
  which is the direct output, no firing; commonly known as ``Readout``, although it's not recommended to literally make
  it the last layer. Internally, it holds ``mem`` as a hidden state and sigmoid-bound ``beta`` as a decay. At each timestep,
  it does the following update: ``mem = mem * beta + mem_delta``. ``mem`` delta can be anything, such as the previous layer's
  output, the synapse's output, the recurrent output; doesn't matter. If no thresholds are present, ``mem`` is made into
  an Exponential Moving Average (EMA) by scaling down ``mem`` delta by ``beta``: ``mem_delta = mem_delta * (1 - beta)``
- ``~B`` suffix stands for ``Binary``, the presence of a threshold, meaning that the layer has 2 possible outputs: a 0 or 1.
  If ``mem`` surpasses the threshold, then the output is a 1, otherwise a 0. We subtract the threshold amount from the neuron that
  fired to reset it back. The threshold is bound by softplus, and is positive.
- ``~T`` suffix stands for ``Ternary``, meaning that the layer has 2 thresholds: a positive and negative one, and thus 3
  possible outputs: -1, 0 or 1. Both thresholds are bound by softplus; and each one accordingly adds/subtracts the threshold
  amount to bring ``mem`` back closer to 0 to reset it.
- ``~S`` suffix stands for ``Scaled``, meaning that the outputs are multiplicatively scaled separately based on their
  polarity, used to make ternary outputs truly independent of each other. This is done with the consideration that after
  the SNN layer, you will have an ``nn.Linear``, or something equivalent. If we don't scale the ternary spikes, then the
  true output of the pair will be ``-w + b``, ``b``, ``w + b``; all equally spaced apart by ``w``, they're not really independent.
- ``D~`` prefix stads for ``Dual``, meaning that all hidden states and parameters are split into a separate positive and
  negative version for greater expressivity and making polarity as a truly separate, alternate signal. The "true" hidden
  states are calculated as the sum of their negative and positive counterparts.
- ``S~`` prefix stands for ``Synaptic``, meaning that before the membrane there is a separate synaptic trace smoothing out
  the inputs over time before they get integrated into the membrane. ``syn`` decays with ``alpha``, and is an EMA.
- ``R~`` prefix stands for ``Recurrent``, meaning that the layer records its own outputs into a separate trace and
  re-integrates it back into the membrane. The trace is ``rec`` which decays with ``gamma``, and is multiplicatively re-integrated
  into ``mem`` via ``rec_weight``.

These combinations result in the following 32 default layers, which sit in ``tracetorch.snn``. These layers also exits
in ``tracetorch.snn.flex`` as their ``LeakyIntegrator`` counterparts, but it's not recommended to use the ``snn.flex``
variants; as it carries all the extra weight of all the extra options which aren't used. ``snn.flex`` is reserved effectively entirely for ``LeakyIntegrator`` for custom experiments.

#. ``LI`` - Leaky Integrator
#. ``DLI`` - Dual Leaky Integrator
#. ``SLI`` - Synaptic Leaky Integrator
#. ``RLI`` - Recurrent Leaky Integrator
#. ``DSLI`` - Dual Synaptic Leaky Integrator
#. ``DRLI`` - Dual Recurrent Leaky Integrator
#. ``SRLI`` - Synaptic Recurrent Leaky Integrator
#. ``DSRLI`` - Dual Synaptic Recurrent Leaky Integrator
#. ``LIB`` - Leaky Integrate Binary fire
#. ``DLIB`` - Dual Leaky Integrate Binary fire
#. ``SLIB`` - Synaptic Leaky Integrate Binary fire
#. ``RLIB`` - Recurrent Leaky Integrate Binary fire
#. ``DSLIB`` - Dual Synaptic Leaky Integrate Binary fire
#. ``DRLIB`` - Dual Recurrent Leaky Integrate Binary fire
#. ``SRLIB`` - Synaptic Recurrent Leaky Integrate Binary fire
#. ``DSRLIB`` - Dual Synaptic Recurrent Leaky Integrate Binary fire
#. ``LIT`` - Leaky Integrate Ternary fire
#. ``DLIT`` - Dual Leaky Integrate Ternary fire
#. ``SLIT`` - Synaptic Leaky Integrate Ternary fire
#. ``RLIT`` - Recurrent Leaky Integrate Ternary fire
#. ``DSLIT`` - Dual Synaptic Leaky Integrate Ternary fire
#. ``DRLIT`` - Dual Recurrent Leaky Integrate Ternary fire
#. ``SRLIT`` - Synaptic Recurrent Leaky Integrate Ternary fire
#. ``DSRLIT`` - Dual Synaptic Recurrent Leaky Integrate Ternary fire
#. ``LITS`` - Leaky Integrate Ternary Scaled fire
#. ``DLITS`` - Dual Leaky Integrate Ternary Scaled fire
#. ``SLITS`` - Synaptic Leaky Integrate Ternary Scaled fire
#. ``RLITS`` - Recurrent Leaky Integrate Ternary Scaled fire
#. ``DSLITS`` - Dual Synaptic Leaky Integrate Ternary Scaled fire
#. ``DRLITS`` - Dual Recurrent Leaky Integrate Ternary Scaled fire
#. ``SRLITS`` - Synaptic Recurrent Leaky Integrate Ternary Scaled fire
#. ``DSRLITS`` - Dual Synaptic Recurrent Leaky Integrate Ternary Scaled fire

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
