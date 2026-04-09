1. What is an SNN?
==================

Spiking Neural Networks (SNNs) are a subtype of RNNs that aim to recreate, or at least more closely follow the
neuron dynamics that we observe in biology, where neurons typically function on discrete signals: all or nothing "spike"
events. SNNs are unfathomably simple in architecture and to understand: each neuron literally collects "energy" over time
and "fires" when it accumulates a certain amount, resetting the amount of energy it has stored up. We call this charge
the "membrane potential": the difference in electrical charge between the interior and exterior of a biological cell.

Do SNNs fully adhere to neuroscience? Also no. Real neuroscience is unfathomably complex if we want to accurately simulate it.
At the tiny scale that biological neurons are, the electrical field of the membrane potential is not uniform throughout
the cell, and this *is* something that has to be accounted for if you want to accurately simulate neuronal dynamics. Our
gold standard for neuron simulation are the Hodgkin-Huxley differential equations, but it should go without saying that
they are incredibly impractical, if not impossible for machine learning.

Instead, SNNs aim to extract the core essence of biological neurons, and discard everything that's not critical. Our goal
is not to blindly copy a working example, because in reality we don't actually know how exactly biological neurons work.
By blindly copying every detail, we risk recreating the Cargo Cult phenomenon. So, what *are* the important aspects of
neuronal dynamics that SNNs copy across?

- Stateful in nature
- Temporal in nature
- Discrete spike events for computation

There are a variety of extra functionality and expressiveness that we can attach, such as recurrence or signal accumulation,
but the core principle remains the same: each neuron accumulates charge until it hits some threshold, fires by sending
the signal downstream, and resets the stored charge amount.

From a research perspective, SNNs are interesting for a variety of reasons:

- **Energy Efficiency**: Working on discrete spikes, we can model them as 1s and 0s, which are incredibly cheap
  multiplicative operations. Human brains consume an astonishingly small amount of power: at most about 20 watts. SNNs
  function on sparsity: nothing happens unless it has to, one would hope that this can bring us closer to the benchmark.
- **Neuromorphic Hardware**: Modern hardware and architecture allow for SNNs to run on Neuromorphic chips where the
  memory and processing happen in the same place, making it far more energy efficient than standard architectures.
- **Biological Plausibility**: With our only working example of general intelligence being biological neurons, it makes
  intuitive sense that there's some kind of reason as to why it's this and nothing else. It would make sense that we try
  re-create a working example first.
- **Convergence of Biology and Technology**: Assuming that we are able to train SNNs (especially online learning),
  it seems promising that Brain-Computer Interfaces (BCIs) or brain implants can run on SNNs.

But that's research, what about the practical angle? Assume that we're not interested in biological plausibility or
energy efficiency and are instead purely focused on actual capability, are SNNs any good? Yes. SNN networks utilize a
mixture of linear functions (such as ``nn.Linear`` or ``nn.Conv2d``) and the nonlinear firing neurons. The nature of the
spikes themselves is what creates the nonlinearity. The simplest of SNN neurons, outputting only 0s (no spike) and 1s
(spike), hence function as logic gates which accumulate information over time. A perfectly trained SNN model will have
"instant" reaction time, not requiring any timesteps to "think" to accumulate charge, and hence are fundamentally no
different from ANNs. But unlike ANNs, SNNs *do* have the innate temporal aspect, they are effectively a natural extension
of what ANNs are. Looking at it from the perspective of mathematical functions, ANNs learn to map timeless functions,
while SNNs learn to map temporal functions.
