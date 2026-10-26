RNN
===

The layers in ``tt.rnn`` keep their hidden states inside the module and process
one timestep per call. Their projections and gates follow the equations below,
while the surrounding model supplies the sequence boundaries and training loss.

Base Layer
----------

.. autoclass:: tracetorch.rnn.Layer
   :members:
   :member-order: bysource
   :show-inheritance:

Layers
------

.. autoclass:: tracetorch.rnn.SimpleRNN
   :members:
   :member-order: bysource
   :show-inheritance:

.. autoclass:: tracetorch.rnn.LSTM
   :members:
   :member-order: bysource
   :show-inheritance:

.. autoclass:: tracetorch.rnn.GRU
   :members:
   :member-order: bysource
   :show-inheritance:
