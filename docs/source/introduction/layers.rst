2. The traceTorch Layers
========================

It is time to (briefly) take a look at the base layer types that traceTorch presents. The practical application of the layers
and when the right one should be used is discussed in more detail in the :doc:`tutorials <../tutorials/index>` section,
here we look at the toggleable features available.

traceTorch provides two main categories of recurrent layers: traditional RNN architectures in ``tracetorch.rnn`` and specialized Spiking Neural Network layers in ``tracetorch.snn``. Detailed explanations of each layer can be found in the :doc:`references <../reference/index>` section.

RNN Layers (tracetorch.rnn)
---------------------------

The RNN module contains traditional recurrent architectures that follow the same ergonomic design principles as the rest of traceTorch:

- ``SimpleRNN``: The classic vanilla RNN with hidden state recurrence
- ``LSTM``: Long Short-Term Memory networks with cell states and gates
- ``GRU``: Gated Recurrent Units with simplified gating mechanisms
- ``Mamba``: Modern State Space Model with selective state spaces

All RNN layers process one timestep at a time and integrate seamlessly with other PyTorch layers, maintaining the traceTorch philosophy of hidden states staying hidden while being easily manageable through the ``tt.Model`` interface.

SNN Layers (tracetorch.snn)
---------------------------

The SNN module follows a slightly unconventional, but incredibly consistent and self-explanatory naming schema, encapsulating a vast array of various neuron mechanics:

- ``LI`` base name stands for ``Leaky Integrator``, the simplest of layer types with just one trace: the membrane potential
  which is the direct output, no firing; commonly known as ``Readout``, although it's not recommended to literally make
  it the last layer. Internally, it holds ``mem`` as a hidden state and sigmoid-bound ``beta`` as a decay. At each timestep,
  it does the following update: ``mem = mem * beta + mem_delta``. ``mem`` delta can be anything, such as the previous layer's
  output, the synapse's output, the recurrent output; doesn't matter. If no thresholds are present, ``mem`` is made into
  an Exponential Moving Average (EMA) by scaling down ``mem`` delta by ``beta``: ``mem_delta = mem_delta * (1 - beta)``
- ``~EMA`` suffix is only used with the ``LI`` type of neurons, and it makes the membrane act as an exponential moving
  average (EMA). This isn't useful in classification where you explicitly train the model return large magnitudes of
  values, but it's useful in other cases where the membrane magnitude need to be stable.
- ``~B`` suffix stands for ``Binary``, the presence of a strictly positive threshold, meaning that the layer has 2 possible
  outputs: a 1 or a 0. ``LIB`` is hence the official name for the ``LIF``.
- ``~T`` suffix stands for ``Ternary``, meaning that the layer has 2 thresholds: a strictly positive and a strictly negative
  one, meaning that the layer has 3 possible outputs: 1, 0 or -1.
- ``~S`` suffix is only used with the ``~T`` suffix to create ``~TS``, which stands for ``Ternary Scaled``, meaning that the
  ternary outputs are multiplicatively separately scaled based on their polarity. This is done so that the three
  possible outputs are truly independent when we consider the downstream layer.
- ``D~`` prefix stads for ``Dual``, meaning that all traces (hidden states) and their decay parameters are split into a
  separate positive and negative version for greater expressivity and unlocking more complex dynamics.
- ``S~`` prefix stands for ``Synaptic``, meaning that before the membrane there is a separate synaptic trace with its
  respective alpha decay that smooth out the inputs over time via an EMA before they get integrated into the membrane.
- ``R~`` prefix stands for ``Recurrent``, meaning that the layer records its own outputs into a separate trace with its own
  gamma decay and re-integrates it back into the membrane in the next timestep. The computation graph is made to work
  even with online learning.

These combinations result in the following 32 default layers, which sit in ``tracetorch.snn``. These layers also exist
in ``tracetorch.snn.flex`` as their ``LeakyIntegrator`` counterparts, but it's not recommended to use the ``snn.flex``
variants; as it carries all the extra weight of all the extra options which aren't used. ``snn.flex`` is reserved effectively entirely for ``LeakyIntegrator`` for custom experiments.

#. ``LI`` - Leaky Integrator
#. ``DLI`` - Dual Leaky Integrator
#. ``SLI`` - Synaptic Leaky Integrator
#. ``DSLI`` - Dual Synaptic Leaky Integrator
#. ``LIEMA`` - Leaky Integrator EMA
#. ``DLIEMA`` - Dual Leaky Integrator EMA
#. ``SLIEMA`` - Synaptic Leaky Integrator EMA
#. ``DSLIEMA`` - Dual Synaptic Leaky Integrator EMA
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

Universal Layer Features
------------------------

When initializing any layer (RNN or SNN), you can also specify the ``dim`` dimension. It's an index as to what dimension the layer should
focus on, which defaults to ``-1``, easily working with MLP networks. Change it to ``-3`` and the layer now works on the color
channel of ``[C, H, W]`` or ``[B, C, H, W]`` tensors. Change it to ``-4`` for example, and the layer works with colored voxel tensors: ``[C, X, Y, Z]``.
Each layer also asks for ``num_neurons``, this value should be the number of values at the desired ``dim`` dimension.
So in the ``dim=-3`` example, it would be the number of color channels.

When initializing any layer, each parameter also optionally asks for the following arguments:

- ``value``: the actual value it should take (instead of ``value``, it's just the name of the parameter, such as ``beta``
  or ``pos_threshold``, et cetera.). Can be a ``float`` for automatic initialization or ``torch.Tensor`` for a custom tensor.
- ``*_rank``: the rank of the parameter. 0 means scalar (per-layer), 1 means vector (per-neuron). Defaults to 1 (vector).
- ``*_learnable``: should the parameter be learnable or static. Defaults to ``True``.