L4. SLIF - Synaptic Leaky Integrate and Fire
============================================

The base ``LIF`` is cool and all, but it loses its signal quickly. Just one firing and ``mem`` has likely lost almost
all of the information that it held before (by the very nature of subtracting the threshold amount, we've close to 0 in
the vast majority of cases). What if we want to hold on the signal for longer?

``SLIF`` does exactly that. Injected current isn't injected directly into ``mem``, but into ``syn``, a second hidden
state, the same size as ``mem``. It is then ``syn`` that injects the current into ``mem``. Which means that while
``mem`` can fire and lose the information, at the next timestep, assuming that ``alpha`` decay allows for it, ``syn``
will still hold a sizeable amount of charge which will allow it to yet again pass that charge into ``mem``. We can thus
(in theory) drastically increase the lifespan of received signals.
