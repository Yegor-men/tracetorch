Biological Plausibility in Machine Learning
===========================================

It is widely believed that biological neural learning most likely largely differs from the modern standard of how to
train AI models. PyTorch is excellent for building and training large, differentiable models; the autograd graph
construction and traversal is, mathematically, the best way to decrease the loss and is certainly the most efficient
solution from a practical driven perspective. However, it's hard to believe that anything remotely similar happens in
real, biological systems like the human brain.

Admittedly, computational neuroscience has still much to uncover, but even from what we can see now, it seems very
unlikely, if not bordering on the impossible, that the brain, let alone each neuron, does anything similar to graph
construction, partial derivatives, or remembering multiple distinct values over an arbitrary amount of time. The
problems only compound when considering the fact that the brain is a recurrent network driven on sparse rewards and
punishments: online learning. Somehow, while doing the forward pass, and when receiving a reward or punishment, the
brain knows how much each neuron or synapse contributed to the result.

traceTorch is not the first to support this kind of learning, but it is designed with simplicity and efficiency in mind.
traceTorch mimics the observed results of biological systems, while using a pseudo-plausible approach for calculating
and updating parameters. After all, the brain is a continually running chemical reaction of unfathomable size, we are
yet to simulate anything remotely similar in magnitude, and hence instead rely on computationally cheap alternatives
centred around being comfortable to use. With a few assumptions and alterations to account for the chemical nature of
the brain, traceTorch isn't too far off from theoretically plausible. It's not a digitized version of biological
neurons, but the goal wasn't that to begin with. The goal is to have a comfortable to use implementation of biological
systems and their abilities in a digital format.