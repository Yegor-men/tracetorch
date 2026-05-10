traceTorch Documentation
========================

.. image:: _static/tracetorch_banner.png
    :alt: traceTorch banner image

traceTorch is a PyTorch library for stateful recurrent layers, built primarily for spiking neural networks.

It keeps model code close to ordinary PyTorch while handling the parts that usually make recurrent and spiking models
awkward: hidden-state initialization, hidden-state resets, truncated history, state persistence, parameter constraints,
and feature dimensions that are not necessarily the last dimension.

The short version is:

* Inherit models from ``tt.Model``.
* Use traceTorch layers inside ordinary PyTorch modules.
* Loop over timesteps yourself.
* Call ``zero_states()`` when a new sequence starts.
* Call ``detach_states()`` when you want online or truncated learning.

Where to go first
-----------------

If you are new to traceTorch, read the documentation in this order:

1. :doc:`Installation <installation/index>` shows how to install the package and run examples.
2. :doc:`Quickstart <quickstart>` builds the smallest useful model and training loop.
3. :doc:`Introduction <introduction/index>` explains the mental model: hidden states, timestep loops, SNN naming, and layer choices.
4. :doc:`Examples <examples/index>` walks through the runnable scripts in the repository.
5. :doc:`Tutorials <tutorials/index>` covers save/load, compile/decompile, and custom layer creation.
6. :doc:`Reference <reference/index>` links the API docstrings.

What traceTorch provides
------------------------

``tt.snn``
    32 leaky-integrator-based SNN layers. The family covers continuous readout layers, binary firing layers, ternary
    firing layers, scaled ternary layers, dual sign-specific traces, synaptic traces, recurrent traces, and combinations
    of those mechanisms.

``tt.rnn``
    ``SimpleRNN``, ``LSTM``, and ``GRU`` layers adapted to traceTorch's state-management style. They process one
    timestep at a time and keep their hidden states internal.

``tt.ssm``
    ``S4``, ``S5``, ``S6``, and ``Mamba``-style layers adapted to traceTorch's recurrent interface. These are useful for
    experimentation and composition with traceTorch layers, but they are not replacements for official optimized SSM
    implementations.

``tt.Model`` and ``tt.Layer``
    The core abstractions. ``tt.Layer`` gives each layer states, constrained parameters, compilation, and dimension
    helpers. ``tt.Model`` recursively manages every traceTorch layer in a model tree.

.. toctree::
    :maxdepth: 2
    :caption: Documentation
    :hidden:

    installation/index
    quickstart
    introduction/index
    examples/index
    tutorials/index
    reference/index
