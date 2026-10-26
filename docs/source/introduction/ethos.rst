Why layers keep their own states
================================

When we classify a whole image at once, a model can calculate its answer from a
single input tensor. A sequence is different: the current input may only make
sense in light of earlier inputs. A spoken digit unfolds over several audio
frames, for example, while the next byte in a sentence depends on the text that
came before it. A recurrent layer keeps some of that history in a hidden state.

We can describe this with a state update and an output:

.. math::

   h_t &= f_\theta(h_{t-1}, x_t), \\
   y_t &= g_\theta(h_t, x_t).

Here :math:`x_t` is the current input, :math:`h_t` is the updated hidden state,
and :math:`\theta` represents the learned parameters. The functions depend on
the layer: a GRU uses gates to combine new input with old memory, while a spiking
neuron can integrate its input into a membrane and produce a firing event.
The state itself may contain several tensors, rather than just one vector.

traceTorch puts that state inside the layer. You call the layer with one input
at a time and receive its output, while the layer retains the values it needs
for the next call. This lets a stateful layer sit inside an ordinary PyTorch
container, such as ``nn.Sequential``, without making every neighboring module
accept or return hidden states.

A timestep comes from the task
------------------------------

For a concrete example, consider a leaky integrator with decay 0.5:

.. code-block:: python

   import torch
   import tracetorch as tt

   layer = tt.snn.LI(1, beta=0.5, learn_beta=False)
   first = layer(torch.tensor([[1.0]]))
   second = layer(torch.tensor([[0.0]]))

Its update is :math:`m_t = 0.5m_{t-1} + x_t`, so the outputs are 1.0 and 0.5.
The second input is zero, but the stored membrane still contains part of the
first input. The caller doesn't need to pass that membrane back into the layer.

We decide what each call represents. In the MNIST tutorial it's a small image
patch; in SHD it's an audio frame; in the language model it's a byte. The library
doesn't assign a physical duration to a call, because the same state update can
be useful at different sampling rates and for sequences without a physical clock.

Keeping the timestep loop in your code also leaves the prediction rule visible.
An image classifier might use only its final output, while a language model
needs a next-byte prediction at every timestep. Both use the same layer interface,
but the point where we calculate loss changes with the task.

The model coordinates memory
----------------------------

A larger network can contain several stateful layers, including ones nested
inside ordinary PyTorch containers. When the network inherits from ``tt.Model``,
we can manage their states together: ``reset_states()`` begins an independent
sequence, ``detach_states()`` preserves memory while limiting gradient history,
and ``save_states()`` lets us resume an unfinished sequence later.

These operations belong in the training or inference code because only that code
knows where a sequence ends. A layer cannot tell whether the next audio frame
continues the current recording or starts a different one. We'll follow those
boundaries in :doc:`stateful_models`, including the difference between forgetting
values and cutting their connection to earlier calculations.

States and parameters play different roles
------------------------------------------

In the leaky integrator, the membrane describes the inputs processed so far,
while the decay describes how the layer behaves for every sequence. The membrane
is a hidden state; the decay is a model parameter. Training can change the decay,
but processing a new input changes the membrane even when we're only evaluating.

That distinction also explains checkpointing. PyTorch's ``state_dict()`` stores
the learned model, whereas ``save_states()`` stores the current sequence memory.
For a new independent input, weights are enough. To continue a particular stream,
we also need its hidden states and the position of its next input.

Combining dynamics with PyTorch modules
---------------------------------------

Built-in SNN layers update each feature's history independently. A linear layer
or convolution supplies the mixing between features, so we can change that
projection without changing the neuron dynamics. GRU and LSTM layers include
their own learned feature projections as part of their gate equations.

All built-in recurrent layers use the final input dimension for features.
For image tensors in channel-first layout, we therefore move the channel axis
explicitly before the recurrent layer and move it back afterward. Keeping that
layout change visible helps us distinguish independent spatial positions from
successive timesteps.

Some neuron parameters also need constraints: a decay stays between zero and one,
for example, while a threshold stays positive. traceTorch stores unconstrained
raw parameters and transforms them when they're accessed. When those weights
remain fixed for evaluation, ``compile_parameters()`` can cache their transformed
values, and ``decompile_parameters()`` restores their intended learnability.
This is separate from ``eval()`` and gradient recording; the :doc:`../tutorials/shd`
tutorial walks through all three operations in context.
