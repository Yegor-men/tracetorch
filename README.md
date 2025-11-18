![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/license/mit)
[![PyPI](https://img.shields.io/badge/PyPI-v0.3.0-blue.svg)](https://pypi.org/project/tracetorch/)

# traceTorch

A highly opinionated, "just works" spiking neural network library for PyTorch.

Written primarily for one person (me). Public because maybe you think the same way.

## What is traceTorch?

traceTorch is a minimalistic SNN library for PyTorch, designed to "just work" out of the box for most use cases, but at
the same time not being restrictive or cumbersome with nonstandard scenarios. It should be simple and intuitive, as if
it was a continuation of PyTorch. With that in mind, traceTorch follows the following "rules":

1. Positive signals are fundamentally excitatory, negative signals are fundamentally inhibitory.
2. Parameters that have to be restricted to some range are done so functionally, no clamping allowed.
3. The same neuron class must be usable for many various architectures and handle reshaping easily.
4. Fixed, scalar, or per-neuron learning mustn't rely on `if` statement branching, should be conceptually clean.
5. Hidden state management should be simple, the user shouldn't have to write extra boilerplate.
6. Tracking of hidden states and parameters must be simple and easily interpretable/usable.

## Installation

traceTorch is a PyPI library, which can be found [here](https://pypi.org/project/tracetorch/).

You can install it via pip. Requirements for the library are listed in `requirements.txt`.

```bash
pip install tracetorch
```

To use, it is recommended to import as such:

```python
import tracetorch as tt
from tracetorch import snn
```

## Documentation and examples

The online documentation can be found [here](https://yegor-men.github.io/tracetorch/). It covers the modules and has a
couple tutorials on how to make your own models. All the tutorials are made to recreate the code found in `examples/`.
They are numbered in the recommended order of completion, the `.py` files are ready to run. It's recommended to (at
least briefly) look at `T0_getting_started.ipynb` to familiarize with the API. Everything you need to make your own
models is in there.

Here's some dummy code as to how traceTorch looks like:

```python
import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

model = snn.Sequential(
	nn.Conv2d(3, 32, 5),
	snn.LIF(32, 0.9, 1.0, view_tuple=(-1, 1, 1)),  # per-channel, 0.9 decay, 1.0 threshold
	nn.Conv2d(32, 64, 5),
	snn.LIF(64, torch.randn(64), 1.0, learn_beta=False, view_tuple=(-1, 1, 1)),  # fixed custom decays
	nn.Flatten(),
	nn.LazyLinear(64),
	snn.LIF(64, 0.5, torch.randn(64), learn_threshold=False),  # Fixed custom thresholds
	nn.Linear(64, 10),
	snn.Readout(10, 0.9, beta_is_vector=False),  # shared scalar decay of 0.9, learnable
	nn.Softmax(-1)
).cuda()

for x, y in loader:
	x, y = x.to(device), y.to(device)
	for _ in range(num_steps):
		out = net(x)  # normal forward, nothing else
	loss = loss_fn(out, y)
	loss.backward()
	optimizer.step()
	model.zero_states()  # resets to None for the next batch
```

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to fork, submit pull requests or report issues, I will occasionally check in
on it.
