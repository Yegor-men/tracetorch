traceTorch
==========

.. image:: _static/tracetorch_banner.png
   :alt: traceTorch banner

A recurrent model uses earlier inputs to make sense of what it sees now.
Spiking neural networks do this with neurons that integrate input, retain a
decaying membrane state and produce firing events. GRUs and LSTMs keep their
own forms of memory, using gates to control what remains from one step to the next.

traceTorch provides these stateful layers for PyTorch. Each layer processes one
timestep and keeps its own hidden states, so we can combine it with ordinary
linear layers, convolutions and containers. Our code supplies the sequence,
decides where memory should reset, and chooses which outputs contribute to loss.

The library includes 32 leaky-integrator variants, SimpleRNN, GRU and LSTM layers,
along with a base class for custom updates and independently shaped states.
We can reset, detach, save or restore those states across a model without passing
them through every module by hand.

Getting started
---------------

If you're new to the library, start with :doc:`installation/index` and the
:doc:`quickstart`. We'll build a small model that remembers a random value
through several timesteps, then train it with a complete PyTorch loop.

:doc:`introduction/index` follows that example with the reasoning behind the
interface and the equations behind the layers. From there, the
:doc:`tutorials/index` work through practical image, audio and text tasks, with
notebooks for exploration and complete Python programs alongside them.

You can use the :doc:`reference/index` whenever you need a method signature,
parameter constraint or the exact update for a particular layer.

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   installation/index
   quickstart
   introduction/index
   tutorials/index
   reference/index
