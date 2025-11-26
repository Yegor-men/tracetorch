L2. The traceTorch Layers
=========================

``traceTorch`` is built around the ``LeakyIntegrator`` superclass (more on that later, in :doc:`L3 <../introduction/L3_leaky_integrator>`).
The base layer is the ``LIF`` - Leaky Integrate and Fire. Internally, it holds the following states/parameters:
 - ``mem``: the membrane which stores the charge
 - ``beta``: the decay factor applied to ``mem`` at each timestep, bound to (0,1) via sigmoid
 - ``pos_threshold``: the threshold that ``mem`` needs to surpass in order to produce a positive spike, bound to (0,inf) via softplus

At each timestep, ``mem`` is updated as such: ``mem = mem * beta + mem_delta``, where ``mem_delta`` is the output of the
previous layer. If a neuron fires by the ``mem`` surpassing the threshold, the threshold amount is subtracted from ``mem`` at that neuron to reset it.

Another base layer is ``Readout``. It's just the ``LIF`` layer minus the thresholds, and an Exponential Moving
Average (EMA) applied on ``mem`` as well to make the magnitude not be affected by the ``beta`` decay. The output is ``mem``, a float, no spikes.
At each timestep, ``mem`` is updated as such: ``mem = mem * beta + mem_delta * (1 - beta)``.

``traceTorch`` presents a wide array of default layers that comfortably work for the vast majority of use cases. The layer
names are modular, various letters meaning various features, absence of a letter means an absence of the feature:
 #. ``S`` - an extra synapse layer ``syn`` before ``mem``. Inputs are accumulated there first, and ``syn`` decays with ``alpha``. It is then ``syn`` that integrates the current into ``mem``
 #. ``R`` - an extra recurrence layer, with a ``weight`` and ``bias`` applied to the previous timestep's outputs, then integrated into ``rec`` which has its own decay ``gamma``
 #. ``B`` - bipolar spiking, outputs are balanced ternary (-1/0/1) instead of binary (0/1). There's now a ``pos_threshold`` and a ``neg_threshold`` bound between (0,inf) and (-inf,0). If ``mem`` surpasses either threshold in the respective magnitude, it sends out a spike of the respective polarity

These combinations result in the following 9 default layers:
 #. ``Readout`` - Readout
 #. ``LIF`` - LIF
 #. ``SLIF`` - Synaptic LIF
 #. ``RLIF`` - Recurrent LIF
 #. ``SRLIF`` - Synaptic recurrent LIF
 #. ``BLIF`` - Bipolar LIF
 #. ``BSLIF`` - Bipolar Synaptic LIF
 #. ``BRLIF`` - Bipolar Recurrent LIF
 #. ``BSRLIF`` - Bipolar Synaptic Recurrent LIF

When initializing any layer, you have to specify the ``dim`` dimension. It's an index as to what dimension the layer should
focus on. Defaulted to ``-1``, it easily works with MLP networks. Change it to ``-3`` and the layer now works on the color
channel of ``[C, H, W]`` or ``[B, C, H, W]`` tensors. Change it to ``-4`` and it works with colored voxel tensors: ``[C, X, Y, Z]``.

When initializing any layer layer, each parameter also asks for the following arguments:
 - ``value``: the actual value it should take. Can be a ``float`` or ``torch.Tensor`` (internal shape assertion)
 - ``rank``: the rank of the parameter. 0 means scalar (shared across the entire layer), 1 means vector (per-neuron) and in the special case of the recurrence ``weight``, 2 is a matrix for all-to-all connectivity
 - ``learnable``: should the parameter be learnable or static

There's some extra arguments unique to decays, thresholds and recurrence tensors, but they're not _that_ critical, and it's
recommended to check the :doc:`documentation <../reference/index>` for that.