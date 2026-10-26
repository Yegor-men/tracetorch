Core
====

The core classes coordinate the pieces used throughout the library. ``Layer``
declares the states and transformed parameters for one update, while ``Model``
finds those layers inside a network so we can manage their memory together.

Model
-----

.. autoclass:: tracetorch.core.Model
   :members:
   :member-order: bysource
   :show-inheritance:

Layer
-----

.. autoclass:: tracetorch.core.Layer
   :members:
   :member-order: bysource
   :show-inheritance:
