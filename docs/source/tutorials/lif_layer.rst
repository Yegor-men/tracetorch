LIF Layer
=========

The Leaky Integrate and Fire (LIF) layer forms the backbone of traceTorch networks, simulating spiking neurons with trace-based updates for biologically plausible learning.

Introduction
------------

Unlike traditional PyTorch layers, the LIF layer is recurrent and uses Leaky Integrate and Fire dynamics to introduce nonlinearity through spiking behavior. It maintains an input trace and membrane potential, enabling gradient approximations without storing full histories.

Mathematics
-----------

At each timestep, the input trace :math:`t` is updated as:

.. math::

   t = t \cdot d + i

where :math:`i` is the input (spike tensor), and :math:`d` is the learnable decay parameter (0 < d < 1).

The membrane potential :math:`m` follows:

.. math::

   m = m \cdot d_m + (i \cdot W)

If :math:`m > \theta` (threshold), the neuron fires (outputs 1), and :math:`m -= \theta`. Here, :math:`W` is the weight matrix, and :math:`d_m` is the membrane decay.

During backward passes, gradients are approximated using average inputs derived from the trace: :math:`i_{avg} = t \cdot (1 - d)`.

Usage Example
-------------

Here's how to use the LIF layer in a simple forward pass:

.. code-block:: python

   import tracetorch.nn as nn
   import torch

   # Create a LIF layer with 10 inputs and 5 outputs
   lif = nn.LIF(in_features=10, out_features=5)

   # Example input: batch of 1, 50 timesteps, 10 features (spikes: 0s and 1s)
   inputs = torch.rand(1, 50, 10) > 0.8  # Sparse spikes
   outputs, states = lif(inputs)  # Outputs are spikes, states include traces/membrane

   print(outputs.shape)  # torch.Size([1, 50, 5])

For recurrent use (e.g., online learning), call the layer multiple times with updated inputs, as it maintains internal states.

See the :py:class:`tracetorch.nn.LIF` API reference for parameter details.

Advantages and Considerations
-----------------------------

- Constant memory: No growth with timesteps.
- Online learning: Backward can be called anytime with a learning signal.
- Trade-off: Gradient approximations may yield lower accuracy than full BPTT.

For more on integration with other layers, see the Sequential tutorial.