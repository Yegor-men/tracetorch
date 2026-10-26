Helpers
=======

Firing functions
----------------

SNN layers use deterministic events with a smooth surrogate derivative by default.
We can instead pass ``smooth`` when the forward output should be a continuous
intensity, or ``stochastic`` when it should sample hard events. These functions
take the membrane's distance from threshold, rather than the raw membrane itself.

.. autofunction:: tracetorch.snn.spike_fn.smooth

.. autofunction:: tracetorch.snn.spike_fn.deterministic

.. autofunction:: tracetorch.snn.spike_fn.stochastic

Initialization transforms
-------------------------

When defining a transformed parameter, we usually want to choose its effective
initial value, such as decay 0.9, rather than its raw representation. These
inverse functions perform that conversion so the forward transform recovers
the intended starting value.

.. autofunction:: tracetorch.inverse_fn.sigmoid

.. autofunction:: tracetorch.inverse_fn.softplus

Decay conversion
----------------

``convert_decay(value, from_type, to_type)`` connects ``"decay"`` with
``"halflife"``, ``"tau"`` and ``"horizon"`` descriptions. Its time quantities
are measured in timesteps, so relating them to seconds also requires the
duration assigned to one call by the task.

.. autofunction:: tracetorch.utils.convert_decay

Layout movement
---------------

.. autoclass:: tracetorch.utils.MoveDim
   :members:
   :show-inheritance:
