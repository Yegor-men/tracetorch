Understanding traceTorch
========================

The :doc:`../quickstart` gives us a working model and training loop. Here we'll
look more closely at the ideas behind that code: what a layer remembers, how
that memory changes over time, and how those choices relate to PyTorch's usual
model and training code.

You can read these pages in order or return to one while working through a
tutorial. We'll start with the library's design, then follow a sequence through
training before looking at the neuron equations and the layers you can use.

.. toctree::
   :maxdepth: 1

   ethos
   stateful_models
   snns
   layers

The equations describe one timestep at a time, just as the code does. Unless we
say otherwise, a subscript :math:`t` means the value after processing input
:math:`x_t`, and states begin at zero when a new independent sequence starts.
