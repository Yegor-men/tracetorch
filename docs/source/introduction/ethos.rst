The Ethos of traceTorch
=======================

traceTorch exists because stateful neural networks should not require messy model code.

Spiking neural networks, recurrent neural networks, and state-space-style models all have the same practical annoyance:
they carry hidden state through time. In plain PyTorch, that often means the model's forward method grows extra
arguments, returns extra tensors, manually initializes state shapes, detaches state at the right time, and threads
hidden values through layers that should otherwise look simple.

traceTorch chooses a stricter rule:

    Layers own their hidden states. Models manage those states.

That one rule shapes the rest of the library.

Hidden states stay hidden
-------------------------

A traceTorch layer accepts one tensor and returns one tensor. It does not ask the caller to pass a membrane, hidden
state, cell state, recurrent trace, or SSM state into ``forward``. Those values live inside the layer.

This keeps model definitions readable:

.. code-block:: python

    self.net = nn.Sequential(
        nn.Linear(784, 128),
        tt.snn.LIB(128),
        nn.Linear(128, 10),
        tt.snn.LI(10),
    )

The code reads like PyTorch because it is still PyTorch. The layer just happens to remember something between calls.

State management stays explicit
-------------------------------

Hidden does not mean uncontrollable. ``tt.Model`` gives you explicit recursive methods:

* ``zero_states()`` starts a new sequence.
* ``detach_states()`` cuts temporal gradients while keeping numerical state.
* ``save_states()`` and ``load_states()`` persist current hidden state.
* ``TTcompile()`` and ``TTdecompile()`` switch constrained traceTorch parameters between trainable and inference forms.

The user decides when a sequence starts, when gradients should flow through time, and when state should be saved. The
library removes boilerplate, not intent.

One timestep at a time
----------------------

traceTorch layers process one timestep per forward call. That is deliberate.

Different tasks disagree about what a sequence means. Some models use the final output. Some average all outputs. Some
accumulate loss at every timestep. Some detach every step for online learning. A hidden sequence API would have to pick
one of those patterns or grow a complicated configuration surface.

traceTorch keeps the loop visible:

.. code-block:: python

    model.zero_states()

    for t in range(num_timesteps):
        output = model(x[t])

This is a small amount of code, and it keeps the training semantics obvious.

Composition over special cases
------------------------------

traceTorch layers are designed to sit next to ordinary PyTorch layers. Put them after ``nn.Linear``, between
convolutions, inside ``nn.Sequential``, or inside your own modules. Use ``dim`` when the feature dimension is not the
last dimension.

This is also why traceTorch includes RNN and SSM-style layers. The library began with SNNs, but the same hidden-state
contract works naturally for GRUs, LSTMs, and recurrent state-space layers.

Opinionated defaults
--------------------

traceTorch defaults are chosen for trainability and clarity.

The most important SNN default is ``quant_fn=nn.Identity()``. A firing layer such as ``LIB`` therefore returns a smooth
firing value by default instead of a hard spike. This makes the default behavior easier to train and closer to a
differentiable dynamical system. If you want discrete forward events, pass a quantizer explicitly.

This is not pretending to be the only correct SNN design. It is traceTorch choosing a stable starting point and making
the sharper choices explicit.

What traceTorch is not
----------------------

traceTorch is not a full training framework. It does not own your dataloaders, losses, logging, checkpoint manager, or
experiment runner.

traceTorch is not an optimized SSM backend. Its SSM layers are adapted to the traceTorch recurrent interface and are
useful for experimentation, but they are not official high-performance sequence-parallel implementations.

traceTorch is not trying to hide PyTorch. The goal is the opposite: keep the model recognizably PyTorch while making
stateful layers feel natural.
