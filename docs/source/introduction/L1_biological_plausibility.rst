1. Biological Plausibility in Machine Learning
==============================================

It is widely believed that biological neural learning most likely largely differs from the modern standard of how we
train artificial neural networks (ANNs). Although neuroscience has much to uncover, even now it seems quite unlikely,
even bordering on the impossible, that the brain and neurons do gradient descent, at least in the conventional sense
and implementation of creating and traversing an autograd graph or something similar. Firstly, such an implementation
implies the following:

#. **Neurons can store an arbitrary amount of data**: In classic backpropagation, we apply the chain rule to a graph
that contains all the instructions to obtaining the loss. By the time we get the loss, we have an exact step-by-step set
of instructions on how we got the loss, and subsequently know exactly what and how to modify. Since the brain is a
recurrent model, this would imply that the brain or neurons can save these "tensors" used in the calculation over an
arbitrary amount of time. All the while this somehow has to be chemically plausible, such that this would happen on an
automated level as a chemical reaction.
#. **The brain has a central control unit**: Not only do the neurons have to keep track of what values were used and how
long ago, but they also have to somehow not use them when conducting backpropagation, instead doing it in a carefully
curated manner, using one value at a time, and then cleaning it from memory, and then proceeding to the next value. It
also can't do anything to the other values until the downstream layer has done the calculations for the learning signal,
so it has to be incredibly well orchestrated on such a large scale.
#. **The brain is on a ridiculous order of magnitude more powerful than the fastest computers**: Not only does it do
backpropagation on arbitrarily long graphs, but it does it so quickly that we don't even notice it happening. One would
think that a sudden change in our "parameters" would make us feel different, briefly lose consciousness, notice
something, anything, but no. But even assuming that we aren't meant to feel any different, the updates happen so
quickly, that by the next instant that the brain gets information it's already updated everything to start building a
new graph.

Now that's not to say that the brain *isn't* doing backpropagation of some kind. At the end of the day, in a simple
implementation, backpropagation can be done with value retrieval, no calculations necessary. Assuming that neurons can
send data not only downstream, but also upstream, it's theoretically possible that they do that, but this still doesn't
tie in with the concept of graph construction and manipulation. We should also consider the other biological limitations
that exist, as well as the discoveries of neuroscience:

#. **The brain is scale invariant**: The concept of scale invariance is that the property of a system or law remains
unchanged even when the scale of measurement are multiplied by a common factor. Fractals are an excellent example; when
presented with some image of a fractal, there is no way to possibly know what scale you reside at, each scale of zooming
looks the same as any other. In the context of brains, this usually refers to the statistical patterns of neural
activity, which also exhibit fractal-esque properties, where signals have the opportunity to potentially travel an
arbitrary distance without being corrupted.
#. **Adding extra neurons should not have any downside**: For context, humans have approximately 86 billion neurons.
But at a certain point in evolution, that number was 0, and it gradually grew. Biologically, things don't happen in
massive steps. We didn't just grow billions of neurons one day, it was slow, gradual, and statistically, animals that
had just a few more neurons were ever so slightly better than the rest and could hence spread their genes. Each neuron
added, is hence providing some kind of tangible bonus, albeit small. Contrast to ANNs where we frequently crash into
overfitting, biological neural masses somehow bypass this issue, and adding extra neurons seems to carry no negative
effect.
#. **Even one neuron should provide some massive, tangible benefit**: Furthering on the concept of evolution, the first,
most rudimentary neuron should have been so significantly better than it's absence that it is the foundation of all
intelligent life on Earth. One single neuron should be able to compute things in a useful enough manner to be viable to
stick around.

Combined together, modern ANN training makes little sense from a biological perspective. It implies a system
unfathomably complex, while also somehow being simple enough so that it could appear through evolution. It's not invalid
to think that we cannot reach general intelligence if the system we use isn't similar to a system we know for sure is
general intelligence. This hence gives rise to biologically plausible artificial neural networks.

The idea of biologically plausible neural networks is that they have the same biological restrictions and ideas to what
we observe happening with neurons and neural masses in real life. The goal isn't necessarily to create a one-to-one
replica (although that is often the interest in computational neuroscience), but to make a framework that could've
potentially evolved. Some frameworks are more biologically aligned than others, after all, we are working with a digital
version here, while real life is an unfathomably complex chemical reaction happening automatically. We ultimately strive
to strike a balance between realism and practicality, to somehow find and keep the best from both worlds.