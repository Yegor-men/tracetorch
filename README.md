![traceTorch Banner](media/tracetorch_banner.png)

# traceTorch

[![License](https://img.shields.io/badge/License-Apache%202.0-purple.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Check out the detailed documentation [here](https://yegor-men.github.io/tracetorch/)

## Roadmap

- Create the poisson click test example
- Implement the trace alternative to REINFORCE
- Make traceTorch into a PyPI library
- Write documentation

## Introduction

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

## What are the advantages of traceTorch?

Like mentioned above, the goal of traceTorch is to make a computationally practical approach to realtime learning based
on sparse rewards. Hence, networks constructed with traceTorch have the following advantages over classic PyTorch ANNs:

- **Recurrent in nature**: Based on SNN principles, traceTorch is recurrent in nature, nonlinearity appearing not
  because of nonlinear functions, but because of the nonlinear nature of the fundamental backbone of SNNs: the Leaky
  Integrate and Fire (LIF) neurons.
- **Constant memory consumption**: As mentioned above, it makes little sense biologically that neurons are somehow able
  to store, traverse and manipulate complex graphs of arbitrary size. Instead, it intuitively makes sense that they
  consume constant memory. traceTorch is built on this principle as well. Despite being recurrent, traceTorch doesn't
  grow its memory consumption regardless of how many forward passes are done.
- **Trace-driven gradients**: traceTorch relies on maintaining an input trace: a running average of the inputs, so that
  it can be possible to reconstruct the "true" average input and hence calculate the approximate output. Thus,
  traceTorch effectively recreates an approximation of the real BPTT autograd graph, without ever needing to actually
  make it or store the history. Through cheap, local updates, traceTorch effectively mimics BPTT by compressing the
  entire history into one singular forward pass, with more value attributed to temporally recent events and inputs.
- **Intermittent, online learning**: With traceTorch being reliant entirely on local calculations, and never needing to
  build a full graph, the backward pass can be invoked at any point in time. Backward passes, reliant on small, local
  per-layer graphs, are also very memory-friendly, consuming a fixed amount of time no matter how many forward passes
  were committed before.

## What are the disadvantages of traceTorch?

traceTorch is not an upgrade or extension to base PyTorch, it is fundamentally designed as a replacement for PyTorch
given very specific, admittedly unusual requirements centred around biological plausibility. Subsequently, you may face
the following challenges:

- **Not a framework to make SNNs**: While traceTorch _could_ be used as a method of making SNNs, if your goal is to
  simply use SNN layers instead of the usual PyTorch layers, traceTorch is not recommended. Libraries such as
  [snntorch](https://github.com/jeshraghian/snntorch) are built specifically around this, they directly incorporate
  SNN layers with the PyTorch autograd in a differentiable manner.
- **Gradient approximations**: The fundamental principle of the backward pass in traceTorch is the reconstruction of
  average outputs based on the learnable parameters and the average input. It is an approximation, not an actual 1:1
  reconstruction or mathematically equivalent substitute. Subsequently, performance will be worse than if using true
  backpropagation.
- **Can't be incorporated directly into an existing classic architecture**: While traceTorch can be used alongside
  classic PyTorch models (such as an autoencoder which can pass it's encoded latent to the traceTorch model), traceTorch
  layers can't be directly interweaved into classic modules like torch.nn.Sequential, the very nature of traceTorch
  making graph free forward passes while being recurrent doesn't consolidate with the classic implementation and
  function of recurrent models which need to store the history.
- **No practical benefit compared to alternatives**: If your goal is to construct an ANN and deploy it, traceTorch is of
  no help. Fundamentally, a simple way to think of the principle of traceTorch is that it compresses a theoretically
  massive autograd graph of multiple recurrent forward passes into just one, all the while doing it on the fly. But
  compression leads to loss, it's simply not possible to gain quality or accuracy by compressing and losing information.

## So why use traceTorch at all?

Admittedly, traceTorch fills quite a specific niche. Online, biologically plausible learning, bordering on the concept
of Single Lifetime Learning (SLL), is hardly something that serves a practical purpose, and is instead designed with the
intent of experiment and research. It is designed for playing around, experimenting. Think of it as a way to make
artificial creatures, and then training them in a similar fashion to how real creatures are trained. To facilitate this,
traceTorch is primarily focused around:

1. A suite of nn modules, akin to the PyTorch counterpart, designed to create the networks.
2. A suite of loss modules, designed to generate the initial learning signal, the derivative of the loss w.r.t. the
   model's outputs.
3. A suite of plot modules, designed to visualize the hidden states and anything else of frequent use or interest.

## Installation

⚠️ WARNING, traceTorch is _not yet_ a library. For now, you'll just have to clone this repository and use the
`tracetorch/` folder within.

```
git clone https://github.com/Yegor-men/tracetorch
cd tracetorch/
pip install -r requirements.txt
```

Then, within a python file where from where the repository root folder is visible, simply do:

```
from tracetorch import tracetorch
```

## Usage examples

There exists `tracetorch/examples/` within which sit test files for playtesting, aimed to test if the components work.

The example files are ready to go files that demonstrate traceTorch in various scenarios. To make sure that you have all
the necessary libraries do:

```
pip install -r examples-requirements.txt
```

## How does traceTorch work?

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

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Acknowledgements

I built traceTorch from the ground up, trying to reverse engineer biological neurons with a sprinkle of intelligent
design, but I would also like to recognize the following projects and people who helped shape my thinking:

- [snntorch](https://github.com/jeshraghian/snntorch) for introducing me to SNN networks in the first place, and their
  principles of function. Ironically, its dependency on constructing the full autograd graph is what largely inspired me
  to make traceTorch.
- [Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov) for introducing me to computational neuroscience, presenting
  interesting concepts in an easy to understand manner. My earliest tests, when I naively wanted to implement 1:1
  biological neurons, largely revolved around his work.
- [e-prop (eligibility propagation)](https://www.biorxiv.org/content/10.1101/738385v4) inspired the whole "trace"
  concept, the idea of keeping a decaying value. Earlier, before traceTorch, I wanted to use e-prop for online learning
  instead. Admittedly unsuccessful in my attempts, and a little put off by the relative difficulty, I instead wanted to
  make something simpler.

## Contributing

Contributions are always welcome. Feel free to submit pull requests or report issues, I will occasionally check in on
it.

You can also reach out to me via either email or Twitter:

- email: yegor.mn@gmail.com
- [Twitter](https://x.com/Yegor_Men)
