5. The advantages of traceTorch
===============================

So, ``traceTorch`` is similar enough to base PyTorch to not be too jarring a difference, but behind the scenes it
updates the parameters in a quite different manner. So, what are the advantages of this approach in contrast to base
PyTorch?

#. **Constant memory consumption**: Since ``traceTorch`` doesn't rely on autograd graphs, or, rather, builds a small
   one during the backward pass to approximate all the forward passes that happened, and updating parameters happens
   consecutively, layer by layer, the memory impact is a fair bit smaller, especially considering the recurrent nature.
#. **Sparse rewards and "online" learning**: Since everything is stored locally and is always ready to do a backwards
   pass regardless of the number of forward passes beforehand, ``traceTorch`` is effectively capable of online learning.
   To be pedantic, it's not exact online learning, since we do not do any extra calculations for the derivatives during
   the forward pass, but that technically makes ``traceTorch`` all the better, since now we do these calculations
   sparsely, only when the backward pass occurs, rather than needing to do them in the forward pass each time to have an
   "instant" backward pass.
#. **Continuous interpolation between stateful and stateless**: By default, ``traceTorch`` layers are initialized to be
   in a recurrent manner. But you can just as easily change the hyperparameters, and disable learning of them in order
   to enforce the model to be stateless. It's a continuous interpolation between maximal recurrence, accumulating
   information over an infinite amount of time, versus rapidly discarding old information.