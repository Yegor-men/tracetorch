L6. The all-powerful, flexible LeakyIntegrator
==============================================

So far, ``RLIF`` has been made to sound like the be all and end all. But, not quite. What if we want ``RLIF`` to have a
``syn`` state and ``alpha`` decay for the previous layer's signals as well? Sure, we can use ``SRLIF``. But what if we
want negative spikes as well, a -1 spike outputted if ``mem`` is below some threshold, and a +1 spike outputted if
``mem`` is above some threshold? What if we want to get rid of the recurrence ``bias`` entirely? What if we want to connect
the recurrence to a different hidden state, and we don't need something as strong as ``rec``? What if we want to make
``mem`` an EMA, while still making it provide spiked outputs? What if we want to make ``mem`` or ``rec`` an EMA too? What
if we don't want spiked outputs whatsoever? There's far too many options, and the naming convention would quickly explode
in combinatorics (easily over 50 distinct "valid" combinations).

Enter the ``LeakyIntegrator`` base class. All the classes that you've seen so far are just light wrappers around it,
a fraction of the true flexibility that's possible. And the ``Leaky Integrator`` can do it all, and allows you to easily
create your own variants and types of neurons that you like, for whatever combination of features it is that you want,
all via an arguably simpler API than with the other pre-made layers. Let's take a look.
