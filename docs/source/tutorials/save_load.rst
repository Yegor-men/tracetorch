Saving and Loading States
=========================

PyTorch already gives you ``state_dict()`` for parameters. traceTorch adds a separate state dictionary for hidden
states: membrane traces, recurrent traces, RNN hidden states, SSM states, and so on.

Why states are separate
-----------------------

Parameters describe the model. Hidden states describe where the model currently is in a sequence.

For many training jobs, you do not need to save hidden states at all. You call ``zero_states()`` for every independent
sequence and train from a fresh state. State saving becomes useful when:

* doing streaming inference;
* checkpointing the middle of a long sequence;
* resuming online learning;
* comparing two models from the exact same hidden state.

Saving
------

Run the model first so states exist. States are lazily initialized, so a freshly constructed model has no tensors to
save.

.. code-block:: python

    model.eval()
    model.zero_states()

    with torch.no_grad():
        for t in range(sequence.size(0)):
            output = model(sequence[t])

    weights = model.state_dict()
    states = model.save_states()

    torch.save(weights, "weights.pt")
    torch.save(states, "states.pt")

The keys in ``states`` are path-like names that point to the layer and state, such as ``"net.2.mem"``.

Loading
-------

Load parameters and states separately:

.. code-block:: python

    model = Net().to(device)
    model.load_state_dict(torch.load("weights.pt", map_location=device))
    model.load_states(torch.load("states.pt", map_location=device), strict=False, device=device)

``strict=False`` is useful when not every state exists yet or when you intentionally load only part of a model's state.
Use ``strict=True`` when you expect the saved state dictionary to exactly match the current model.

Shape checking
--------------

If a layer already has a state tensor and the loaded state has a different shape, traceTorch raises a ``ValueError``.
This prevents silent state corruption when batch size, target dimension, or model structure changed.

Continuing a sequence
---------------------

After loading states, continue calling the model normally:

.. code-block:: python

    model.eval()
    with torch.no_grad():
        next_output = model(next_timestep)

Do not call ``zero_states()`` after loading unless you intentionally want to discard the loaded states.
