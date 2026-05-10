Functional
==========

``tt.functional`` contains small helper functions used by traceTorch layers and useful in experiments.

Decay Helpers
-------------

.. autofunction:: tracetorch.functional.halflife_to_decay

.. autofunction:: tracetorch.functional.decay_to_halflife

.. autofunction:: tracetorch.functional.timesteps_to_decay

.. autofunction:: tracetorch.functional.decay_to_timesteps

Parameter Transforms
--------------------

.. autofunction:: tracetorch.functional.sigmoid_inverse

.. autofunction:: tracetorch.functional.softplus_inverse

.. autofunction:: tracetorch.functional.mamba_scale

Spike Functions
---------------

.. autofunction:: tracetorch.functional.sigmoid4x

Quantizers
----------

.. autofunction:: tracetorch.functional.round_ste

.. autofunction:: tracetorch.functional.stochastic_round_ste

.. autofunction:: tracetorch.functional.probabilistic_ste
