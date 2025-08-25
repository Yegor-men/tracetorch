![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-purple.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI](https://img.shields.io/badge/PyPI-v0.3.0-blue.svg)](https://pypi.org/project/tracetorch/)

``traceTorch`` is a PyTorch-based library that
implements [eligibility propagation](https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf),
replacing the PyTorch default backpropagation through time with lightweight, per-layer eligibility traces, enabling
biologically inspired, constant time and memory consumption learning on arbitrarily long or even streaming sequences.

## Documentation

It is highly recommended that you read the [documentation](https://yegor-men.github.io/tracetorch/) first. It contains
the following sections:

1. **What is Eligibility Propagation?**: Covers the background, theory and intuition for eligibility propagation.
   It's not crucial to know, but it helps.
2. **Getting Started**: Explains the installation, setup and usage patterns of ``traceTorch``.
3. **Tutorials and Examples**: Walks through various practical examples and implementations that utilize ``traceTorch``.
   The resultant code can be found in ``tutorials/``.
4. **API Reference**: Detailed documentation of all modules, classes and functions that ``traceTorch`` contains.

## Origins & Acknowledgements

I originally developed ``traceTorch`` with the intent of exploring biologically inspired, constant-memory learning for
spiking neural networks (SNNs). The idea was that each layer maintains an input trace, from it could be reconstructed
the average input, and from that, by reusing the layer's parameters: the average output. Layer by layer, the backward
pass would happen, passing the learning signal from one layer to the next, each one reconstructing the average outputs
and utilizing PyTorch's autograd to compute derivatives from the approximated output based on the incoming learning
signal. Effectively, instead of calling .backward() on an arbitrary size autograd graph, ``traceTorch`` allowed to
"compress" it into one forward pass, and subsequently in theory required only one backward pass. Call an arbitrary
number of forward passes, in the backward pass it would approximate the "average" forward pass, and then do a real
.backward() pass on effectively one timestep.

In itself, the approach was fine. Models could learn and generalize. Works also started on a realtime alternative to
REINFORCE, aptly named REFLECT, as it would strengthen or weaken the chain of choices that led up to a reward, rather
than those that, as a consequence of them occurring, led to a reward. Mathematically effectively similar, simply adapted
to online learning.

However, further testing revealed that doing a backward pass for each timestep, rather than one at the end, drastically
improved the model's ability to learn, being almost as fast as traditional backpropagation. However, this came at the
cost of an immense decrease in speed, as each timestep would effectively do: 1 real forward pass and in the backward
pass effectively redo the forward pass and do an actual autograd .backward() pass inside. Subsequently, I got interested
if it was possible to "pre-bake" the backward pass graph, after all, the graph was the same each time. The sole reason
autograd was used was because the real underlying function is complex. However, biologically, through evolution, it's
not difficult to imagine that the correct function got pre-baked into the neurons, that in reality, it's only a matter
of retrieval of values, without any computations involved.

I hence found myself back at the starting
point: [e-prop](https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf). Eligibility propagation,
which originally I did not understand, inspired the very input trace mechanics. It effectively uses this "pre-baked"
concept, and, hence, training by doing a backward pass at each timestep is not only faster (inference speed, not
necessarily training speed) while also being cleaner.

Hence, ``traceTorch`` is now focused around eligibility propagation. The old code can still be found and used in
``legacy/``. I may or may not return to it sometime, from the perspective of sparse .backward() calls.

### Acknowledgements

- **[snntorch](https://github.com/jeshraghian/snntorch)**: For introducing me to spiking neural networks and practical
  SNN tooling.
- **[Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov)**: For accessible presentations on computational
  neuroscience
  that influenced my thinking about spiking dynamics and simple, interpretable neuron models.
- **[E-prop / eligibility propagation](https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf)**: The
  very paper that powers most of ``traceTorch``.

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
