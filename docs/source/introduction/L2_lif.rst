L2. LIF - Leaky Integrate and Fire
==================================

The simplest of neurons, the LIF (Leaky Integrate and Fire). No temporal dynamics, nothing complicated, it simply
accumulates charge into the hidden state ``mem`` and fires once the charge surpasses a threshold. Upon firing a spike,
we subtract the threshold amount from ``mem`` in order to reset the neuron and ready it to fire again. At each timestep,
before integrating the injected current into ``mem``, we also decay it by a decay value ``beta`` bound between 0 and 1.