Creating a Custom Layer
=======================

A traceTorch layer is a PyTorch module with a small amount of state-management structure. This tutorial recreates a
minimal GRU-like layer to show the moving parts without drowning in SNN details.

The goal
--------

We want a layer that:

* stores a hidden state ``H``;
* creates ``H`` lazily from the input shape;
* works on any feature dimension through ``dim``;
* integrates with ``tt.Model.zero_states()`` and ``detach_states()``.

Subclass a traceTorch layer
---------------------------

For RNN-style layers, subclass ``tt.rnn.Layer``. It already inherits from ``tt.Layer``.

.. code-block:: python

    import torch
    from torch import nn
    import tracetorch as tt


    class MiniGRU(tt.rnn.Layer):
        def __init__(self, in_features: int, out_features: int, dim: int = -1):
            super().__init__(num_neurons=out_features, dim=dim)

            self._initialize_state("H")
            self.gates = nn.Linear(in_features + out_features, 2 * out_features)
            self.candidate = nn.Linear(in_features + out_features, out_features)

Registering a state
-------------------

``_initialize_state("H")`` records the state name and sets ``self.H = None``. That is enough for ``tt.Model`` to find and
manage the state later.

Forward pass
------------

The forward pass has the same shape as the built-in layers:

.. code-block:: python

    def forward(self, x):
        self._ensure_states(x)

        x = self._to_working_dim(x)
        H = self._to_working_dim(self.H)

        H_x = torch.cat([H, x], dim=-1)
        reset, update = torch.sigmoid(self.gates(H_x)).chunk(2, dim=-1)

        candidate = torch.tanh(self.candidate(torch.cat([H * reset, x], dim=-1)))
        H = H * (1 - update) + update * candidate

        self.H = self._from_working_dim(H)
        return self.H

``_ensure_states(x)``
    Creates ``H`` if it is ``None``. The state shape matches ``x`` except the target dimension becomes ``out_features``.

``_to_working_dim(x)``
    Moves the configured ``dim`` to the last dimension so linear layers can operate normally.

``_from_working_dim(H)``
    Moves the last dimension back to the configured ``dim``.

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
    model.zero_states()
    y = model(torch.rand(16, 64))
    print(y.shape)
    # torch.Size([16, 32])

Adding constrained parameters
-----------------------------

SNN and SSM layers often need constrained parameters. For example, a decay should stay between zero and one. SNN layers
use helper methods such as ``_register_decay``:

.. code-block:: python

    class DecayLayer(tt.snn.Layer):
        def __init__(self, num_neurons: int, beta: float = 0.9):
            super().__init__(num_neurons)
            self._initialize_state("mem")
            self._register_decay("beta", beta, rank=1, learnable=True)

        def forward(self, x):
            self._ensure_states(x)
            self.mem = self.mem * self.beta + x
            return self.mem

The public ``self.beta`` value is activated through a sigmoid, so it stays in ``(0, 1)``. The raw stored parameter is
``self.raw_beta``. This is what allows ``TTcompile()`` and ``TTdecompile()`` to optimize constrained parameters later.

Checklist
---------

When creating a custom traceTorch layer:

* call the superclass with ``num_neurons`` and ``dim``;
* call ``_initialize_state`` for every hidden state;
* call ``_ensure_states(x)`` before using states in ``forward``;
* use ``_to_working_dim`` before operations that expect features last;
* write updated states back through ``_from_working_dim``;
* return one tensor, keeping hidden states internal.
