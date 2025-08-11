How does traceTorch work?
=========================

Fundamentally, traceTorch asks the following question:

_"Assuming we somehow get a weighted average of the inputs prioritizing recent events, what was the average output the
model produced, and hence, combined with the learning signal, how should we adjust the layer's parameters?"_

Intuitively, this makes sense. The moment of doing a backward pass is really considered as the moment the model gets a
reward / punishment. It's fair to assume that the most recent events had the most influence on this outcome, because
otherwise, we'd get the reward / punishment earlier. So really, it's a testament to say that the earliest events and
actions had far less to do with the outcome than the most recent events an actions.

So, in a hypothetical thought experiment. Imagine we are observing the `n`th layer, and we want to update the parameters
for it. In classic backpropagation, we would look at our autograd graph, recall the outputs of this `n`th layer and
knowing the learning signal: the derivative of the loss w.r.t. to this layer's outputs, we know which direction we'd
want to tweak this layer's outputs. Subsequently, we can calculate the derivative of the loss w.r.t. the layer's
parameters using the chain rule, and hence update them. We also use the chain rule to calculate the derivative of the
loss w.r.t. to the inputs, and hence get the learning signal that we send upstream (to the `n-1`th layer), as the inputs
to the `n`th layer are also the outputs of the `n-1`th layer. And hence, we can split the entire backward pass into
a chain of local calculations.

Now, looking at our case with traceTorch. Recall that we somehow saved the weighted average input: this is the input,
but averaged over time to favor more recent events. The downstream layer (the `n+1`th layer in the forward pass)
constructs the learning signal, and the downstream layer has its own recording of the average inputs, so we can be sure
that the learning signal is sensible. Next, we just need to know the average output over this same timeframe. We could
keep track of the average outputs the same way that we do with the inputs, and if traceTorch is the way things happen
in biological neurons, then they probably do, but here, computationally, we have no idea how exactly the parameters
affected the loss. In reality, it's some function, a complex one at that, but why calculate it by hand when we can cheat
by using autograd? We will simply approximate the outputs based on the input

More concretely, how does this look in pseudocode? For a start, let's take a look at how we obtain this "average input".
Take note of the following terminology for the input trace calculations:

- `t`: the input trace
- `d`: a learnable parameter for the input trace decay, some value bound between (0, 1)
- `i`: the input at any given timestep

At each timestep, we do the following to modify the input trace:

```
t = t * d + i
```

Simple enough. Mathematically, this is equivalent to as if we saved all the inputs from each timestep and then
multiplied each one by `d**(how long ago)`. The trick is to simply reverse engineer the input, assuming we have a trace
that's unchanging. So, we are effectively asking: _"Let's say our trace isn't changing over time, what input would we
need to have on average in order to get this observed trace?"_

```
t * d + i = t
t * d + i - t = 0
t * (d - 1) + i = 0
i = -t * (d - 1)
i = t * (1 - d)
```

And hence, that is the average input `i`. Considering that traceTorch works with SNNs, the average input is really the
firing frequency of this input. Hence now, using this input we have to calculate the average output, the firing
frequency of the output neuron. Let's take a look at the workhorse LIF neuron and how it works to understand how we do
this.

Take note of the following terminology for the LIF neuron calculations:

- `i`: the raw input (since it's SNNs, it's spikes, i.e. a tensor with only 1s or 0s)
- `W`: a learnable parameter, the weight matrix
- `m`: the membrane level, effectively the amount of charge the neuron is currently holding
- `d`: a learnable parameter for the membrane level decay, some value bound between (0, 1)
- `t`: a learnable parameter, the threshold that the membrane level has to surpass in order to fire and produce an
  output (a 1, whereas no output is 0), bound to be a positive number

At each timestep we do the following to calculate the LIF layer's outputs:

```
synaptic current = i * W
m *= d
m += synaptic current
should fire = m > t
m -= t * should fire
```

But this isn't differentiable, nor the backward pass. This is the forward pass, the actual outputs we have. Recall how
we kept track of the input trace. Looks awfully familiar to the membrane state, does it not? The only difference is that
we occasionally subtract the threshold level amount if the neuron does fire. So let's treat the membrane level as
another trace, and use the same mathematics here. Since we have the average input `i` calculated from `i=t(1-d)` during
the input trace calculations, we can also calculate the average synaptic current by simply passing it through the weight
matrix `W` as if it were a normal forward pass.

```
average synaptic current = average input * W
```

Now, we can treat the average synaptic current as the average input. This time around though we are interested in what
value the trace will stabilize at given we have the average input and decay. We can just rearrange the formula from
before to make `t` the trace the subject, and replace the input trace terms for the respective membrane trace terms:

```
i = t * (1 - d)
i / (1 - d) = t
t = i / (1 - d)
m = average synaptic current / (1 - d)
```

Now, recall that what we are really trying to do is calculate the average output, or technically the firing frequency.
There _technically is_ a clean formula derived from this, since the membrane level is really a differential equation,
but testing revealed this mathematically exact equation to be numerically unstable, while also having a limited domain.
So instead let's think from the first principles, and some things we can easily deduce:

1. If the threshold `t` is higher than `m = average synaptic current / (1 - d)`, then that means that even if we had
   infinite time, the membrane level will _never_ reach the firing threshold, so the firing frequency is `0`
2. If the `average synaptic current` is higher than the threshold `t`, then it really doesn't matter what `d` or `m` or
   anything is, because we will surpass the threshold at each timestep, so the firing frequency is `1`.

It's not hard to imagine a sigmoid function stretching across from these two points to smoothly approximate the firing
frequency in between, as well as beyond the domain, and that is what traceTorch does behind the scenes. In reality, the
sigmoid function isn't exactly `0` or `1` at these points; and the decay `d` for both the input trace and membrane decay
is actually stored as a raw value with an infinite domain, just passed through sigmoid before using; and so is the
threshold `t` stored as a raw value, but then passed through softplus; there are more layers than just the LIF; but
these are minor details, this explanation is sufficient to explain the works of traceTorch: the goal of any traceTorch
layer is to try approximate the average outputs from the average inputs in some smooth, differential way.