L5. RLIF - Recurrent Leaky Integrate and Fire
=============================================

``SLIF`` is much better, a well worth upgrade over ``LIF``. We can now hold on to signals for far longer, theoretically
for any arbitrary time, given that we've the proper ``alpha`` and ``beta`` decay combination.

But that's not enough. It's still a stateless system, in the SNN sense. Sure, we can hold on to the signal for a _very_
long time, but we will inevitably lose it. What if we don't want to?

``RLIF``, the Recurrent Leaky Integrate and Fire does exactly that. It records the previous timestep's outputs, and
re-integrates them at each timestep back into the layer to get _far_ more complex, stateful dynamics. To make matters
better, the signals processed by the recurrent layer pass through a ``weight`` and ``bias``, as a dense linear layer
re-integrates the signals, and saves them to ``rec``, which has its own ``gamma`` decay as to not mix signals from the
recurrence and the previous layer. It is with ``RLIF`` that we can finally achieve truly stateful dynamics, neurons
firing, or their absence of firing, triggering complex cascades for themselves and the other neurons in the layer.
