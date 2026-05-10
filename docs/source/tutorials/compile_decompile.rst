Compiling and Decompiling
=========================

Some traceTorch parameters are stored in an unconstrained raw form and transformed when used. Decays, for example, are
stored as raw values and passed through a sigmoid so the effective decay stays in ``(0, 1)``. Thresholds are passed
through a softplus so they stay positive.

This is useful for training, but a trained inference model does not need to recompute those transforms every forward
pass. ``TTcompile()`` bakes the transformed values into buffers. ``TTdecompile()`` restores the raw trainable form.

Basic use
---------

.. code-block:: python

    model = Net().to(device)

    # train normally
    ...

    model.eval()
    model.TTcompile()

    with torch.no_grad():
        output = model(x)

After compilation, parameters registered through traceTorch's parameter helpers are no longer trainable raw parameters.
Use compiled models for inference, not training.

Returning to training
---------------------

.. code-block:: python

    model.TTdecompile()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

``TTdecompile()`` uses each parameter's inverse transform to recreate the raw parameter or buffer.

What gets compiled
------------------

Only parameters registered through traceTorch's registration helpers are compiled. Ordinary PyTorch parameters, such as
``nn.Linear.weight``, are unaffected.

For an SNN layer, this usually includes values such as:

* ``beta`` / ``alpha`` / ``gamma`` decays;
* thresholds;
* biases;
* some scale or recurrent-weight parameters, depending on the layer.

State and compilation are separate
----------------------------------

Compilation changes parameter storage, not hidden state values. You can compile a model with or without existing hidden
states.

For clean inference, a common pattern is:

.. code-block:: python

    model.eval()
    model.zero_states()
    model.TTcompile()

    with torch.no_grad():
        for t in range(sequence.size(0)):
            output = model(sequence[t])

Checking equivalence
--------------------

Compiled and decompiled models should produce the same outputs for the same weights and same initial states. A minimal
sanity check is:

.. code-block:: python

    model.zero_states()
    baseline = model(x)

    model.TTcompile()
    model.zero_states()
    compiled = model(x)

    assert torch.allclose(baseline, compiled)

    model.TTdecompile()
    model.zero_states()
    decompiled = model(x)

    assert torch.allclose(baseline, decompiled)
