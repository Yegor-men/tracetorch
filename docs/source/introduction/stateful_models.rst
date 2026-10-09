Stateful Models
===============

traceTorch layers are stateful. A layer such as ``tt.snn.LIB`` stores a membrane trace. A layer such as ``tt.rnn.GRU``
stores a hidden state. Custom layers can store vector or matrix states. The caller still sees a normal PyTorch layer:
one tensor in, one tensor out.

The rule is simple: hidden states stay inside the layer, but the model can manage them.

Why inherit from ``tt.Model``?
------------------------------

``tt.Model`` is a thin extension of ``nn.Module``. It recursively walks through the model tree and finds traceTorch
layers, even if they are inside ``nn.Sequential`` or nested modules.

.. code-block:: python

    class Net(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(784, 128),
                tt.snn.LIB(128),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

That model now has recursive state methods:

``reset_states()``
    Sets all traceTorch states to ``None``. The next forward pass recreates them with the correct shape, dtype, and
    device.

``detach_states()``
    Detaches all current states from the computation graph. This is used for online learning or truncated
    backpropagation through time.

``save_states()``
    Returns a dictionary of current hidden states.

``load_states()``
    Loads a state dictionary created by ``save_states()``.

``compile_parameters()`` and ``decompile_parameters()``
    Compile and decompile constrained traceTorch parameters so inference can skip repeated activation transforms.

Timestep loops
--------------

traceTorch layers process one timestep per forward call. The library does not hide the sequence loop because different
tasks want different loss accumulation, readout, and detach behavior.

Full sequence backpropagation:

.. code-block:: python

    model.reset_states()
    running_loss = 0

    for t in range(num_timesteps):
        output = model(x[t])
        running_loss = running_loss + loss_fn(output, target[t])

    running_loss.backward()
    optimizer.step()

Online or truncated learning:

.. code-block:: python

    model.reset_states()

    for t in range(num_timesteps):
        output = model(x[t])
        loss = loss_fn(output, target[t])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.detach_states()

The first version keeps the computation graph through time. The second version cuts temporal gradients after each
timestep while preserving the numerical state values.

When to reset
-------------

Call ``reset_states()`` when a new independent sequence starts. In a dataloader, that usually means once per batch.

.. code-block:: python

    for sequence, label in dataloader:
        model.reset_states()
        for t in range(sequence.size(0)):
            output = model(sequence[t])

Do not call ``reset_states()`` inside the timestep loop unless you intentionally want to erase temporal memory.

Lazy state shapes
-----------------

States start as ``None``. Each state declares its own trailing shape and negative slice boundary:
``reference.shape[:dim] + state_shape``. By default ``dim=-1`` replaces the final input dimension;
``dim=None`` appends the state shape without removing dimensions.

For example, an SNN state of shape ``(32,)`` with input ``[B, H, W, 32]`` has the same shape
as the input. A custom state of shape ``(32, 4)`` would instead have shape ``[B, H, W, 32, 4]``.
All leading dimensions represent independent instances; the caller handles time iteration externally.
Reset before changing the instance shape.

Built-in layers work feature-last. For channel-first images, use ``tt.utils.MoveDim(-3, -1)``
before the layer and ``tt.utils.MoveDim(-1, -3)`` after it. Move the model to its target device/dtype
before allocating or loading hidden states, which are ordinary tensor attributes.

Saving states
-------------

Model parameters and hidden states are separate. Use ``state_dict()`` for parameters and ``save_states()`` for current
hidden traces.

.. code-block:: python

    torch.save(model.state_dict(), "weights.pt")
    torch.save(model.save_states(), "states.pt")

    model.load_state_dict(torch.load("weights.pt"))
    model.load_states(torch.load("states.pt"), strict=False)

State saving is useful for streaming inference, checkpointing a long sequence, or resuming an online model without
forgetting its recent history.
