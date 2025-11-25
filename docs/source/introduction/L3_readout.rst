L3. Readout
===========

Spiked outputs are cool and all, but we often want a floating point. We could apply a function atop the spikes, but
that's a remedy, not a solution. So how about we don't reset ``mem`` after firing because we won't fire at all?

The ``Readout`` layer is a variant of ``LIF`` that doesn't fire. It has no threshold, and as output returns ``mem`` at
each timestep. For the sake of computational stability, ``mem`` is also changed to be an exponential moving average to
make it invariant to the ``beta`` decay.
