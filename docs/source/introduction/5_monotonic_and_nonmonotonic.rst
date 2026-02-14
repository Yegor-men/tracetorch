5. Monotonic and Nonmonotonic Functions
=======================================

The task of any neural network is to learn to map some high dimensional function. In the simplest of cases, it can be a
literal x -> y coordinate mapping, say for a sine wave. Higher in complexity would be to memorize some image: x and y
coordinate mapped to a pixel brightness, or multiple values for an RGB image. The task of decoder-only transformer
LLMs is to predict the next token based on the current sequence of tokens (considering positional embedding, which is effectively
a representation of time). In similar vain, the task for any SNN (as for any RNN) is to learn a function that explicitly
occurs over time: a temporal function, preferably without any hints as to what moment in time it's in.

In mathematics, functions have a property of being monotonic. It is a function defined on an ordered set, and in simple
terms, it constantly moves in one direction: either ever increasing or ever decreasing; a one-to-one function, if you may.
Nonmonotonic functions are subsequently those that do not ever increase or ever decrease; a many-to-one function, if you may.
Hence, ``y=ln(x)`` is a monotonic function, while ``y=sin(x)`` is not. However, nonmonotonic functions can be split into
a set of monotonic ones. In the case of ``y=sin(x)``, we can split it on ``x=pi/2``, ``x=3pi/2``, ``x=5pi/2`` and so on.
Each segment is monotonic: only ever increasing or ever decreasing.

For the case of SNNs, we can look at data being monotonic or nonmonotonic from the temporal lens. The question of monotonicity
is effectively: "is the goal to blindly accumulate charge?". This isn't always a simple yes or no, much more likely a mix,
but it largely determines the difficulty of the task at hand, and with it, the architecture that your SNN model needs to be.

Let's take a look at the staple MNIST task. The most familiar approach to all would be to pass it through a CNN, have a
kernel learn to detect features, and based on the features that it has detected, learn to classify the image. But in itself,
this task has no aspect of time. We certainly could feed it to an SNN based CNN model (or even a simple MLP), and it will
learn, considering that without the time aspect SNNs are really no different from an MLP, but this really isn't a temporal
function. So let's make it. There's two things that we can do:

#. Rate code the image for each timestep, and feed the entire image to the model at each timestep for some amount of timesteps, expect the correct classification at the end
#. Flatten the 28x28 image into a sequence of 784 timesteps and feed in one pixel at a time, expect the correct classification at the end

The latter of these approaches undoubtedly sounds more difficult than the former, because it is. But why? The first approach,
even though we rate code the image (instead of feeding float values, we take a ``torch.Bernoulli`` sample of the image,
thus making it significantly more noisier, as if simulating that the input came from an upstream SNN layer), is nothing more
than blind accumulation. Say that we set the brightest pixels which used to be a 1.0 to a 0.05, meaning that each pixel
that's part of the digit only has a 5% to actually fire at each timestep, and maybe for complexity just add noise by tuning
the pixels that used to be 0.0 to a 0.01 to thus naturally add noise. Is the task more difficult? Of course. But say that
we just blindly accumulate this charge over 100 timesteps, what do we expect to see? The dullest pixels will have fired an
average of 1 time, the brightest ones would have fired an average of 5 times. It's faint, but very distinct. All the model
has to do is to turn that 1:5 scale back to 0:1, and now our task is no different from the original MNIST. At the very core,
there is no real complexity in the task, it is entirely confined by the question of if your network can learn to classify
MNIST images in the first place, because it can certainly with relative ease undo the noise. Thus this task is monotonic:
simple blind accumulation of charge.

Now what about flattening the image into a sequence of 784 timesteps and feeding one pixel at a time, row by row from the
top left one down to the bottom right one. If the model blindly accumulates charge, will anything work? Likely: 1s have less
pixels than other digits. But a 4 and 5 will likely have around the same amount, as will the rest of the digits. No, in order
to be able to differentiate the digits now, the model will have to explicitly rely on the specific sequence that they fire in.
Heavy conditionals based on the order. With 784 timesteps, and a 0/1 on each one, we technically have 1.017458e+236 possible
combinations of firing patterns available, and the task of the model is to learn a high order rule that somehow maps these
784 timesteps to one of 10 classes. The task is made more difficult by the fact that the model doesn't get the pixel's coordinate
at each timestep, which means that it has to create it's own sense of time and pacing to that it figures out where the pixel
is relative to the zeroed states that the model started with. This makes the task nonmonotonic: blind accumulation simply
doesn't work.

Or, we *could* feed in the coordinate alongside the value at each timestep, but in that case we'd have to shuffle the order
in which we show the pixels. Now, the task of the model is to effectively create a blank canvas for the image, and based
on the coordinate and value it gets, paint in one pixel at a time. Is this task monotonic or nonmonotonic? A mix. On one hand,
the task of simply storing the canvas is monotonic: here you need to only blindly accumulate charge in order to re-create the
original MNIST image. But the task of actually drawing in the pixel, figuring out what the coordinates actually map to in
the DHS is a whole different challenge.

In reality, almost every task is a mix of monotonic and nonmonotonic in nature. SNN neurons are monotonic in their very
nature: they have to accumulate charge in order to fire and send a signal downstream. Even the most complex of layer types
are fundamentally monotonic: accumulate charge to fire. But the overall task is most likely nonmonotonic in nature, but
it relies on separate monotonic segments to do the conditional work. For example, in the case of feeding coordinateless
values for 784 timesteps, some parts of the model are most certainly going to learn to accumulate charge: some pixel will
only fire if the model accumulates enough data and charge to think that the image is a 1 for example. From this angle, you
can raise the argument that the *true* task of any SNN model is not so much to learn a temporal function, but to learn to
decompose a temporally nonmonotonic function into a set of distinct monotonic conditionals.

What does this mean from the practical perspective? Monotonic segments are significantly simpler to learn than nonmonotonic
ones. In the case of MNIST, a rate coded model can most certainly learn with a uniform ``beta`` decay throughout all layers,
you likely don't even need ``beta`` to be a vector (per-neuron) parameter, it can just be a scalar (per-layer), and everything
will work out. But for nonmonotonic tasks, such as the 784 long sequence, such a model will *significantly* struggle. You could
approach the task strategically: proclaim that since it's a 784 long sequence, then the model needs an appropriate decay
value such that it can remember for 784 steps: the decay hence should be ``1 - 1/784=0.998724489796``, but even then, you
will struggle heavily. Assume that it really is that simple. But what about if you have ``alpha``, and not just ``beta``?
What if you have recurrence to ease the requirements on the decays, and now you've ``gamma``? What about the other parameters?
Reality is, you likely do not know the "true" timescale that the model works on, let alone each neuron in the net. The model
likely suffers a lot from rigid initialization, and instead, the strategy should be a lot simpler, following the KISS principle
as does traceTorch in general: Keep It Stupid Simple. Parameters allow to be initialized as ``torch.Tensor``, which means
that for decays and thresholds you could literally use ``nn.rand(num_neurons)`` for initializations, to simply scrape through
the configurations available. Worry not about the "temporal window" argument. The expected largest value for a random uniform distribution
of ``n`` samples is ``(n+1)/n``, and if we conver that decay through the temporal window formula ``t=1/(1-d)``, we get
that for ``n`` neurons in a layer, and that the decays follow a uniform distribution between 1 and 0, the largest effective
timescale will be 1 more timestep than the number of neurons. Neat. To nail in the point, this is how the ``snn.BSRLIF`` layer
initialization looks like in the example byte level language model:

 .. code-block:: python

    snn.DSRLITS(
        hidden_dim,
        pos_alpha=torch.rand(hidden_dim),
        neg_alpha=torch.rand(hidden_dim),
        pos_beta=torch.rand(hidden_dim),
        neg_beta=torch.rand(hidden_dim),
        pos_gamma=torch.rand(hidden_dim),
        neg_gamma=torch.rand(hidden_dim),
        pos_threshold=torch.rand(hidden_dim),
        neg_threshold=torch.rand(hidden_dim),
        pos_scale=(torch.randn(hidden_dim) + 1.0),
        neg_scale=(torch.randn(hidden_dim) + 1.0),
        pos_rec_weight=torch.rand(hidden_dim),
        neg_rec_weight=torch.rand(hidden_dim),
    )
