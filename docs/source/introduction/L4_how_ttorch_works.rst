4. How does traceTorch work?
============================

Now that we know what SNNs are, how traces work, we can finally look at how and why ``traceTorch`` works. Recall that
the fundamentals of backpropagation with autograd and graphs is that we want to find out how much each learnable
parameter affects the loss. If we have our entire graph that leads to some final loss value being calculated, we can
easily traverse the graph and apply the chain rule along the way.

The first important thing to understand is that the big autograd graph can actually be split into many smaller graphs,
each one for each layer, and hence turn this one big operation into a number of simpler, local ones. For this to work,
each layer should store the inputs it got, as well as the computation graph used to produce the outputs. Then, when the
downstream layer sends a learning signal (the derivative of the loss w.r.t. this layer's outputs) to this layer,
we can easily traverse our small local graph to find out how much each parameter affected the output, and since we know
how wrong the outputs are, we know how wrong the parameters are. Calculate the derivative of the incoming learning
signal w.r.t. the input and we have constructed the learning signal that we will send upstream. Repeat until all layers
are done.

The second thing to understand is that recurrent models are very memory heavy. Imagine a simple recurrent model with 2
layers that runs for 100 timesteps. Not only will we need to save the 100 raw inputs, but we need to save the 100
intermediate tensors going from layer 1 to 2, and the 100 outputs. We are effectively eating 100 times the amount of
memory that we would have with a normal model. This is a very big issue with backpropagation through time (BPTT), so you
can say goodbye to training anything for a decently long amount of time, you will have to resort to the truncated
version of BPTT: TBPTT. But, even then, you can still bid goodbye to anything of decent size or for long time sequences.
You are hardware limited by the amount of things you can store, you simply cannot find patterns over long periods of
time if you don't have those tensors recorded in any way.

At the very core, ``traceTorch`` effectively converts the entire recurrent history of model inference with n timesteps
into a single timestep. Internally, each layer maintains an input trace, and then at the time of doing a backwards pass,
calculates the average input ``i`` needed to receive the recorded trace. Then, since we now have an average input, the
average firing frequency of the upstream neurons, we can recalculate and approximate the average output, the average
firing frequency of the output neurons. Now, we have a small, local graph that we can traverse and find out how much
each parameter affected the learning signal.

Another benefit of this approach, aside from constant memory consumption, is the fact that it theoretically allows to
spot those super long range patterns. The input trace could have a decay at 0.999, thus valuing long range thinking, or
it could have some small value like 0.5, thus valuing short term observations. It's not hard to see how, since the LIF
layer is really just a logic gate of sorts, the deeper into the network you go, the more it's focused on spotting
meta-patterns, the patterns of patterns. For this, it's going to need to be centred around long term observation.

In fact, the charge level (technically called the membrane level, inspired by biology) in the LIF neuron is also a
trace. We decay it by some value at each timestep, to it add the current (called the synaptic current, also because of
biology). The only difference is that we occasionally reset the membrane level, but other than that it's the same trace.
Subsequently some of the layers present in ``traceTorch``, such as ``tracetorch.nn.LIS()``, which outputs a
softmax probability distribution rather than spikes per se, also utilize the trace formulae to find at what levels the
membrane will stabilize, and hence what the average output is.
