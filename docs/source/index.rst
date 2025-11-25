traceTorch Documentation
========================

.. image:: _static/tracetorch_banner.png
   :alt: traceTorch banner image

traceTorch is bult around a single, highly compositional neuron superclass, replacing the restrictive "layer zoo" of
countless disjoint neuron types with the ``LeakyIntegrator``. This design encapsulates a massive range of SNN dynamics:

- synaptic and recurrent filtering
- rank-based parameter scoping for scalar, per-neuron or matrix weights
- optional Exponential Moving Average (EMA) on any hidden state
- arbitrary recurrence routing to any hidden state
- flexible polarity for spike outputs: positive and/or negative

All into declarative configuration on one class. By abstracting this complexity, traceTorch provides both the robust
simplicity required for fast prototyping via familiar wrappers (``LIF``, ``RLIF``, ``SLIF``, ``Readout``, etc.) and the
unprecedented flexibility required for real research.


.. toctree::
   :maxdepth: 2
   :caption: The docs cover the following:

   introduction/index
   tutorials/index
   reference/index
