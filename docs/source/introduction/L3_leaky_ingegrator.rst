L3. The all-powerful, flexible LeakyIntegrator
==============================================

As mentioned before, all ``traceTorch`` layers are children of the ``LeakyIntegrator`` superclass. They easily cover
the vast majority of use cases, but at the same time, aren't too flexible for experimentation. For example:
 - what if you want ``syn`` or ``rec`` to be an EMA too?
 - what if you want the reccurence ``weight`` and ``bias`` to connect to a different state?
 - what if you want the complexity of synaptic and recurrence, but you don't want spiked outputs?

And so on. The ``LeakyIntegrator`` superclass allows you to do it all via a quite simple API. You can easily create your
own layers with your own configurations by just copying the existing wrappers. Funnily enough, the API is arguably even
even simpler. Let's take a look.