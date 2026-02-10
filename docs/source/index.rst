traceTorch Documentation
========================

.. image:: _static/tracetorch_banner.png
   :alt: traceTorch banner image

traceTorch is bult around a single, highly compositional neuron superclass, replacing the restrictive layer zoo of
countless disjoint neuron types with the ``LeakyIntegrator`` superclass. This design encapsulates a massive range of SNN
dynamics:

- Flexible polarity for spike outputs: positive and/or negative or none at all for a readout layer
- Optional synaptic and recurrent signal accumulation into separate hidden states
- Rank-based parameter scoping for per-layer (scalar) or per-neuron (vector) parameters, learnable or static
- Optional Exponential Moving Average (EMA) on any hidden state

All into declarative configuration on one class using sensible, powerful defaults.

By abstracting this complexity, traceTorch provides both the robust simplicity required for fast prototyping via
familiar wrappers and the unprecedented flexibility required for real research and models. In total, traceTorch presents
a total of 12 easy to use layer types which directly integrate into existing PyTorch models and API: ``LIF``, ``BLIF``,
``SLIF``, ``RLIF``, ``BSLIF``, ``BRLIF``, ``SRLIF``, ``BSRLIF``, ``Readout``, ``SReadout``, ``RReadout``, ``SRReadout``;
with an API simple enough that you can add more with little effort.

This documentation is written to help you familiarize with SNNs and traceTorch. By the end of it, you should have all
the background knowledge necessary to understand and make your own networks, as well as a solid understanding of the library
and good practices for creating powerful models. It is recommended to at least briefly read the :doc:`introduction <../introduction/index>`
section to familiarize yourself with some background theory and terminology before proceeding with the library.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction/index
   installation/index
   tutorials/index
   reference/index
