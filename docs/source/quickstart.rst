Quickstart
==========

Let's build a small model that remembers a number. We'll give it a random value
at the first timestep, follow that with seven zeros, and ask it to reproduce the
original value from its final output. That gives us a complete training example
without needing to download a dataset.

The last input is always zero, while the answer depends on the first input, so
the model has to carry information between calls. We can use a leaky integrator
for that memory, with a linear layer on either side to project into and out of
eight hidden features:

.. math::

   u_t &= W_{in}x_t + b_{in}, \\
   m_t &= \beta m_{t-1} + u_t, \\
   \hat y_t &= W_{out}m_t + b_{out}.

Here :math:`x_t` is the input at timestep :math:`t`, :math:`m_t` is the stored
membrane, and :math:`\hat y_t` is the prediction. The decay :math:`\beta`
controls how much of the previous membrane remains. We'll use
:math:`\hat y_8` to predict :math:`x_1`; the intermediate outputs don't contribute
to this task's loss. The matrices :math:`W_{in}` and :math:`W_{out}` are the
linear layers' learned weights, and :math:`b_{in}` and :math:`b_{out}` are their
biases.

We don't need an extra activation function here because recalling a number is a
linear task. LI provides memory, while its update remains linear for a fixed
decay. In the practical tutorials, firing functions provide nonlinear
updates when the task needs them.

The complete example
--------------------

.. literalinclude:: ../../examples/quickstart/main.py
   :language: python

The input tensor has shape ``[8, 16, 1]``: eight timesteps, sixteen independent
sequences, and one feature. Only its first timestep contains random values; the
remaining timesteps are zeros. ``targets`` keeps a copy of those first values,
which we'll compare with the model's final predictions.

Each forward call receives one timestep. ``LI`` keeps its membrane between calls,
and inheriting from ``tt.Model`` lets us reset that memory across the model.
We reset before each repetition because it begins a new sequence; otherwise the
model would also see memory left over from the previous training update.

Within each update, we clear the parameter gradients, process the eight inputs
in order, and calculate mean squared error from the last prediction. Calling
``backward()`` follows the calculation through all eight timesteps, so the
optimizer can adjust both the projections and the membrane decay. Resetting
states and clearing gradients serve different purposes, which is why both appear
in the loop.

Trying it
---------

After :doc:`installation/index`, run the example from the downloaded repository:

.. code-block:: bash

   python examples/quickstart/main.py

With seed 0, the loss falls from about 2.784 to less than 0.000001, and the final
predictions approach the first inputs. We deliberately reuse a small batch here
to make the training procedure easy to follow; this is a demonstration of
temporal memory, rather than a dataset benchmark.

You can see why the reset belongs outside the timestep loop by moving it inside
that loop for an experiment. The model then forgets the initial value before it
sees the final zero, so it can only learn a shared answer for all sixteen targets.
Move the reset back before continuing with the other examples.

If you'd like to understand how these states and decays work, continue with
:doc:`introduction/index`. The :doc:`tutorials/index` use the same training ideas
for image, audio and text tasks.
