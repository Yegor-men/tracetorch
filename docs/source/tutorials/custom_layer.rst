Creating a Custom Layer
=======================

A traceTorch layer extends ``nn.Module`` with state management and optional transformed parameters.
The base constructor takes no feature count or dimension. A layer processes one timestep per call;
the caller loops over time.

Declaring states
----------------

Each state declares its own trailing shape and how many trailing reference dimensions to replace:

.. code-block:: python

    import torch
    from torch import nn
    import tracetorch as tt


    class MiniGRU(tt.rnn.Layer):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.define_state("H", (out_features,))
            self.gates = nn.Linear(in_features + out_features, 2 * out_features)
            self.candidate = nn.Linear(in_features + out_features, out_features)

        def forward(self, x):
            self.zero_states(x)
            H = self.H
            reset, update = torch.sigmoid(self.gates(torch.cat([H, x], dim=-1))).chunk(2, dim=-1)
            candidate = torch.tanh(self.candidate(torch.cat([H * reset, x], dim=-1)))
            self.H = H * (1 - update) + update * candidate
            return self.H

``define_state(name, shape, dim=-1)`` sets the attribute to ``None`` and records its allocation rule.
``zero_state(name, x)`` allocates zeros only if the state is ``None``, using
``x.shape[:dim] + shape`` and the reference dtype/device.

For example, input ``[B, L, D]`` and state shape ``(512, 4)`` give ``[B, L, 512, 4]`` with
``dim=-1`` or ``[B, 512, 4]`` with ``dim=-2``. ``dim=None`` removes no input dimensions.
Zero and positive dimensions are rejected. Scalar state shapes ``()`` are also supported.

Reset states before changing the instance shape or starting a new independent sequence.
``reset_state(name)`` and ``reset_states()`` set states to ``None``; detach methods cut their history.
Custom layers can override ``zero_state`` when they need a parameter-dependent resting state rather than zeros.
The plural method calls the singular method for every declared state.

Using the layer
---------------

.. code-block:: python

    class Net(tt.Model):
        def __init__(self):
            super().__init__()
            self.layer = MiniGRU(64, 32)

        def forward(self, x):
            return self.layer(x)


    model = Net()
    model.reset_states()
    y = model(torch.rand(16, 64))
    # y.shape == torch.Size([16, 32])

Built-in layers use the last dimension for features. For channel-first images, place
``tt.utils.MoveDim(-3, -1)`` before a layer and ``tt.utils.MoveDim(-1, -3)`` after it in
``nn.Sequential``. Hidden states remain in the feature-last layout.

Defining parameters
-------------------

The base layer accepts an actual tensor of any shape, without scalar expansion or rank restrictions:

.. code-block:: python

    self.define_parameter(
        "beta",
        torch.full((512,), 0.9),
        learnable=True,
        initialization_fn=tt.inverse_fn.sigmoid,
        activation_fn=torch.sigmoid,
    )

The initialization function runs once, converting the effective value into its raw representation.
``self.raw_beta`` is always an ``nn.Parameter``. ``self.beta`` returns the activated raw value or,
while compiled, its detached cached value. No initialization function is retained for decompilation.

Use ``set_parameter_learnable("beta", False)`` to change intended learnability. This setting survives
the temporary freeze imposed by compilation. Directly changing ``requires_grad`` does not update that intention.

SNN convenience helpers
-----------------------

SNN scalar/per-neuron construction belongs to ``tt.snn.Layer`` rather than the generic base:

.. code-block:: python

    class DecayLayer(tt.snn.Layer):
        def __init__(self, num_features: int, beta: float = 0.9):
            super().__init__()
            self.num_features = num_features
            self.define_state("mem", (num_features,))
            self.define_decay("beta", beta, rank=1, learnable=True)

        def forward(self, x):
            self.zero_states(x)
            self.mem = self.mem * self.beta + x
            return self.mem

The SNN helpers ``define_decay``, ``define_threshold`` and ``define_neuron_parameter`` support numeric
scalar expansion: rank 0 shares one value; rank 1 initializes a value per neuron. Tensor inputs
are used as scalar or per-neuron values directly.
