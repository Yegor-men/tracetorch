![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-purple.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI](https://img.shields.io/badge/PyPI-v0.3.0-blue.svg)](https://pypi.org/project/tracetorch/)

``traceTorch`` is a PyTorch-based library built on the principles of spiking neural networks, replacing the PyTorch
default backpropagation through time with lightweight, per-layer input traces, enabling biologically inspired, constant
time and memory consumption learning on arbitrarily long or even streaming sequences.

## Documentation

It is highly recommended that you read the [documentation](https://yegor-men.github.io/tracetorch/) first. It contains:

1. **Introduction**: An introduction to traceTorch, how and why it works, it's founding principles. It's thoroughly
   recommended that you read through the entire introduction and gain an intuitive understanding before proceeding.
2. **Tutorials**: Various tutorials to create your own traceTorch models. The resultant code can be found in
   `tutorials/`, complete with plotting and display of any useful metrics.
3. **Documentation**: The actual documentation to all the modules included in `traceTorch`. It includes detailed
   explanations, examples and math to gain a full understanding of how ``traceTorch`` works behind the scenes.

## Related work & acknowledgements

I built ``traceTorch`` from the ground up with the goal of exploring biologically inspired, constant-memory learning for
spiking networks. Many projects and papers shaped the ideas here — the following helped the most and deserve
acknowledgment.

### Acknowledgements

- [snntorch](https://github.com/jeshraghian/snntorch) — for introducing me to spiking neural networks and practical SNN
  tooling. The design choice in snntorch to build full autograd graphs was a helpful contrast that inspired
  ``traceTorch``’s constant-memory approach.
- [Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov) — for accessible presentations on computational neuroscience
  that influenced my thinking about spiking dynamics and simple, interpretable neuron models.
- [E-prop / eligibility propagation](https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf) — the
  idea of maintaining decaying eligibility traces and combining them with modulatory signals heavily inspired the
  “trace” abstraction in ``traceTorch``. While e-prop aims at approximating full RTRL, ``traceTorch`` focuses on a
  lighter-weight single-lifetime learning pipeline using local traces to obtain the average input and subsequently
  output for entirely local, small graph backpropagation.
- Reward-modulated plasticity / three-factor rules — the biological and theoretical literature on reward-modulated STDP
  and three-factor learning (local eligibility × global reward) shaped the REFLECT concept: keep a lightweight trace and
  apply credit via a scalar reinforcement signal.

### How traceTorch is different

``traceTorch`` sits at the intersection of these ideas but with a different engineering emphasis:

- Single-Lifetime Learning (SLL): the API and algorithms are designed to learn online during a single continuous run
  through the data/environment with constant memory usage (no BPTT or replay buffers).
- Constant-memory trace mechanics: each layer maintains compact decaying traces (inputs, outputs, and log-prob traces)
  that approximate time-averages; these traces are used to build a tiny differentiable window at the time of update
  rather than building a long computational graph.
- Practical policy-gradient for SNNs (REFLECT): a trace-based REINFORCE-style estimator that keeps an averaged log-prob
  trace of sampled actions and uses it to produce low-variance, correct learning signals for spiking layers.
- Modular, pluggable design: lightweight LIF/LIS layers, Reflect learning modules, and a Sequential orchestration
  layer make it easy to build SNNs that learn online while remaining debuggable and serializable (state_dict friendly).

If you’re curious about specific papers: look into e-prop (Bellec et al.), eligibility traces and three-factor
learning (Frémaux & Gerstner), and reward-modulated STDP literature (Izhikevich, Florian). These influenced the ideas
here and are useful starting points if you want more theory.

## Roadmap

- Finish REFLECT
- Create the poisson click test example
- Make .grad function as a trace
- Make dynamic LR
- Finish writing the documentation
- Move tutorial code to separate repository
- Implement abstract graph based models, not just sequential

## Installation

``traceTorch`` is a PyPI library, which can be found [here](https://pypi.org/project/tracetorch/).

You can install it via pip. All the required packages for it to work are also downloaded automatically.

```
pip install tracetorch
```

To import, you can just do ``import tracetorch``, although more frequently it will look like this:

```
import tracetorch as tt
from tracetorch import snn
```

## Usage examples

`tutorials/` contains all the tutorial files, ready to run and playtest. The tutorials themselves can be found
[here](https://yegor-men.github.io/tracetorch/tutorials/index.html).

The tutorials make use of libraries that ``traceTorch`` doesn't necessarily use. To ensure that you have all the
necessary packages for the tutorials installed, please install the packages listed in `tutorials/requirements.txt`

```
cd tutorials/
pip install -r requirements.txt
```

It's recommended to use an environment that does _not_ have ``tracetorch`` installed if using the tutorials,
``tracetorch/`` is structured identically to the library, but is of course a running release.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to submit pull requests or report issues, I will occasionally check in on
it.

You can also reach out to me via either email or Twitter:

- yegor.mn@gmail.com
- [Twitter](https://x.com/Yegor_Men)
