Following a sequence through the model
======================================

A layer keeps its states until we change them, so the first question in a
stateful model is where one sequence ends and another begins. We'll use image
patches and text streams to see how that choice affects both the numerical memory
and the calculations PyTorch keeps for training.

Independent sequences and batches
---------------------------------

In the MNIST example, an image becomes 49 patches. Those patches belong to one
sequence, so we keep the states while reading them. The next batch contains
different images, which means its first patch needs fresh states:

.. code-block:: python

   for sequence, labels in dataloader:
       model.reset_states()
       for timestep in sequence:
           output = model(timestep)
       loss = loss_fn(output, labels)

The reset sets declared states to ``None``. On the next forward call, the layer
allocates zeros using that input's leading dimensions, device and data type.
This also lets the last batch be smaller than the others: we create the right
number of state positions when we see its first input.

Each batch position has its own memory, even though every position uses the same
learned parameters. If position 0 previously contained one image and now contains
another, keeping its membrane would mix the two images' histories. Sequence
boundaries therefore concern which input occupies each position, as well as the
shape of the tensor.

Memory and gradients are connected, but different
-------------------------------------------------

Consider a leaky integrator that starts from zero:

.. math::

   m_t = \beta m_{t-1} + x_t,
   \qquad
   m_T = \sum_{t=1}^{T}\beta^{T-t}x_t.

The final membrane contains a weighted history of the inputs. It also shows why
a loss at the final timestep can train earlier calculations: for a fixed decay,

.. math::

   \frac{\partial m_T}{\partial x_t} = \beta^{T-t}.

PyTorch records the calculations that connect those values. Calling
``loss.backward()`` follows them through the timesteps to calculate parameter
gradients; this is backpropagation through time. We can use the final output for
a classification task or include predictions from every timestep for a language
model, without changing the layer's state interface.

As a sequence grows, retaining all those calculations also uses more memory.
For a long text stream, we can keep the numerical state while limiting how far
gradients follow its history. The language tutorial uses an update at every
timestep:

.. code-block:: python

   model.reset_states()
   for timestep in range(streams.shape[0] - 1):
       optimizer.zero_grad(set_to_none=True)
       output = model(streams[timestep])
       loss = loss_fn(output, streams[timestep + 1])
       loss.backward()
       optimizer.step()
       model.detach_states()

After ``detach_states()``, the next call begins with the same state values,
but its loss won't send gradients back through the preceding timesteps.
This is one-step truncated backpropagation through time: past information can
affect the prediction, but later losses cannot train the earlier operations
that stored it. We can instead process several steps before backward and detach
when we want gradients to follow a longer interval. By contrast,
``reset_states()`` discards the values themselves, so it removes the passage's
numerical context too.

The inputs need to continue the same streams in the same batch positions for
this to make sense. Shuffling unrelated passages while retaining states would
attach each passage to someone else's history. The :doc:`../tutorials/language_model`
tutorial constructs continuous stream lanes and works through this boundary.

Calculating a loss doesn't clear the graph or detach the states. Backward
normally frees the graph's saved intermediates, but retained state tensors
still refer to their preceding calculations until we detach them. This matters
when an optimizer changes parameters before the next timestep.

Clearing gradients is another separate operation. ``optimizer.zero_grad()``
clears accumulated parameter gradients before an update, but it neither resets
nor detaches hidden states. That is why it appears alongside state operations
in the training code.

Choosing state shapes
---------------------

A state doesn't have to be a vector. Each declaration specifies its own trailing
shape, and the default allocation rule replaces the input's last feature axis
with that shape. For a feature count :math:`F` and a history count :math:`K`,
we can therefore give a layer both a membrane vector and a matrix of histories:

.. list-table::
   :header-rows: 1

   * - Input
     - Declared state
     - Allocated state
   * - ``[B, F]``
     - ``(F,)``
     - ``[B, F]``
   * - ``[B, F]``
     - ``(F, K)``
     - ``[B, F, K]``
   * - ``[B, H, W, F]``
     - ``(F, K)``
     - ``[B, H, W, F, K]``

Here :math:`B` is batch size, and :math:`H` and :math:`W` are spatial dimensions.
Every leading position has independent states; none of these axes represents
time. Time still comes from the outer loop. The :doc:`../tutorials/custom_layer`
tutorial gives the matrix a concrete purpose by storing several filtered
histories for each neuron.

For channel-first images, we move channels to the final dimension before a
built-in recurrent layer. ``MoveDim`` provides an ordinary module for that
layout change:

.. code-block:: python

   layer = nn.Sequential(
       tt.utils.MoveDim(-3, -1),
       tt.snn.LIB(32),
       tt.utils.MoveDim(-1, -3),
   )

The surrounding model sees ``[B, 32, H, W]``, while the neuron sees and stores
states in ``[B, H, W, 32]`` layout. This makes each spatial location an independent
state position; a convolution can mix neighboring positions when the task needs it.

We also move the model to its intended device and data type before beginning
a sequence. Hidden states are plain tensor attributes, so moving a model after
it has started processing inputs doesn't automatically move those existing states.
For an independent sequence we can reset first; for a continuation we can load
the saved states onto the destination model.

Saving an unfinished sequence
-----------------------------

When we save a model, its weights describe how it processes any sequence. Hidden
states describe the particular sequence it has already seen, so we save them
separately when we need to continue that sequence:

.. code-block:: python

   torch.save({"weights": model.state_dict()}, "model.pt")
   torch.save({"states": model.save_states(), "cursor": next_position}, "states.pt")

``save_states()`` returns detached copies, which lets the live model continue
without changing that saved snapshot. To resume, we construct the same model,
load its weights, load its hidden states, and continue from the saved input
position. Resetting after loading would discard the very history we're restoring.

For independent images or recordings, the next input starts from zero anyway,
so a hidden-state file isn't needed. The practical examples keep ``model.pt``
and, where useful, ``states.pt`` together in their local ``checkpoints/`` folder.
The language-model tutorial compares every remaining output with an uninterrupted
run, so we can check that the restored state represents the same computation.
