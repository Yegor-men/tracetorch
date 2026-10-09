Compiling and Decompiling
=========================

Transformed parameters store raw weights and apply an activation when accessed. For example,
a decay is stored as a raw value and exposed through a sigmoid. Compilation caches the activated
value for inference, avoiding that transform on each forward pass.

Basic use
---------

.. code-block:: python

    model.eval()
    model.compile_parameters()
    model.reset_states()

    with torch.no_grad():
        for t in range(sequence.size(0)):
            output = model(sequence[t])

A compiled parameter retains its raw ``nn.Parameter`` but is temporarily frozen, with its gradient
cleared. The activated cache is a detached, cloned, nonpersistent buffer. It moves with the module
but is not included in ``state_dict()``. Its presence determines whether access uses the cache.

Ordinary PyTorch parameters such as ``nn.Linear.weight`` are unaffected. Compilation does not
disable autograd for the rest of the model; use ``torch.no_grad()`` for inference.

Returning to training
---------------------

.. code-block:: python

    model.decompile_parameters()
    model.train()

Decompilation deletes caches and restores each parameter's intended learnability, including
parameters configured not to learn. Raw parameter objects are not replaced, so existing optimizers
remain valid. No inverse transform is required.

Use ``layer.set_parameter_learnable(name, bool)`` to update intended learnability. When a parameter
is compiled it remains frozen, and the new setting takes effect on decompilation. Repeated
compilation and decompilation are safe no-ops.

Individual parameters
---------------------

Layer methods have singular and plural forms:

.. code-block:: python

    layer.compile_parameter("beta")
    layer.decompile_parameter("beta")
    layer.compile_parameters()
    layer.decompile_parameters()

Model methods use only the plural forms, recursively calling the layer operations.

Saving and loading
------------------

Compiled or not, save weights with ``torch.save(model.state_dict(), path)``. Only raw weights and
ordinary persistent PyTorch buffers are included, not compiled caches or hidden states.
Loading a weight state dictionary invalidates existing caches and restores intended learnability;
compile again if desired. Learnability settings come from model configuration, not the weight file.

Saving the entire Python model with ``torch.save(model, path)`` is different from saving a weight
state dictionary and does not provide these cache/state exclusions.

Checking equivalence
--------------------

.. code-block:: python

    model.reset_states()
    baseline = model(x)

    model.compile_parameters()
    model.reset_states()
    compiled = model(x)
    assert torch.allclose(baseline, compiled)

    model.decompile_parameters()
    model.reset_states()
    decompiled = model(x)
    assert torch.allclose(baseline, decompiled)

Compilation does not change hidden states. Reset them only when a fresh sequence is intended.
