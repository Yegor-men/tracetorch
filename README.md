![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/license/mit)
[![PyPI](https://img.shields.io/badge/PyPI-v0.9.2-blue.svg)](https://pypi.org/project/tracetorch/)

# traceTorch

A strict, ergonomic, and powerful Spiking Neural Network (SNN) library for PyTorch.

traceTorch is bult around a single, highly compositional neuron superclass, replacing the restrictive layer zoo of
countless disjoint neuron types with the `LeakyIntegrator` superclass. This design encapsulates a massive range of SNN
dynamics:

- Flexible polarity for spike outputs: positive and/or negative or none at all for a readout layer
- Optional synaptic and recurrent signal accumulation into separate hidden states
- Rank-based parameter scoping for per-layer (scalar) or per-neuron (vector) parameters, learnable or static
- Optional Exponential Moving Average (EMA) on any hidden state

All into declarative configuration on one class using sensible, powerful defaults.

By abstracting this complexity, traceTorch provides both the robust simplicity required for fast prototyping via
familiar wrappers and the unprecedented flexibility required for real research and models. In total, traceTorch presents
a total of 12 easy to use layer types which directly integrate into existing PyTorch models and API: `LIF`, `BLIF`,
`SLIF`, `RLIF`, `BSLIF`, `BRLIF`, `SRLIF`, `BSRLIF`, `Readout`, `SReadout`, `RReadout`, `SRReadout`; with an API simple
enough that you can add more with little effort.

## Why traceTorch?

Existing SNN libraries often feel restrictive or require verbose state management. Aside from the technical features and
capabilities, traceTorch follows a fundamentally different philosophy, revolving around ergonomics and usability:

- **Architectural Flexibility**: All existing traceTorch layers are just small wrappers of the `LeakyIntegrator`
  superclass, and it's incredibly easy to add your own alterations and combinations of the features you like.
- **Automatic State Management**: No need to manually pass hidden states through `.forward()`, each layer manages its
  own hidden states, and calling `.zero_states()` on a traceTorch model recursively clears _all_ the hidden states the
  entire model uses, no matter how deeply hidden they are. In a similar style, `.detach_states()` detaches the states
  from the current computation graph.
- **Lazy Initialization**: Hidden states are initialized as `None` and allocated dynamically based on the input shape.
  This completely eliminates "Batch Size Mismatch" errors during training and inference.
- **Dimension Agnostic**: Whether you are working with `[Time, Batch, Features]` or `[Batch, Channels, Height, Width]`
  tensors, layers _just_ work. Change a single `dim` argument during layer initialization to indicate the target
  dimension the layer acts on. Defaults to `-1` for MLP, `-3` would work for CNN (channels are 3rd last in
  `[B, C, H, W]` or `[C, H, W]`). The tensors are automatically move the target dimension to the correct index so that
  the layers work.
- **Smooth Constraints**: Decay and threshold parameters are constrained via Sigmoid and Softplus respectively. No hard
  clamping means that gradients flow smoothly and accurately everywhere.
- **Rank Based Parameters**: Instead of messy flags like `*_is_vector` or `*_is_scalar`, traceTorch uses a single
  `*_rank` integer to define each parameter scope: 0 for a scalar (per-layer), 1 for a vector (per-neuron).
- **Sensible, Powerful Defaults**: traceTorch defaults to learnable, per-neuron (rank 1) parameters for flexibility and
  EMA on synaptic and recurrent traces for numerical stability; because real research and real models thrive on
  heterogeneity. Overridable if you want, but sensible defaults means less boilerplate.

## Installation

traceTorch is a PyPI library found [here](https://pypi.org/project/tracetorch/). Requirements for the library are listed
in `requirements.txt`. Take note that examples found in `examples/` may have their own requirements, separate from the
library requirements.

```bash
pip install tracetorch
```

If you want to run the example code without installing the PyPI package, or alternatively want to edit the code
yourself, you should install traceTorch as an editable install.

```bash
git clone https://github.com/Yegor-men/tracetorch
cd tracetorch
pip install -e .
```

## Quick Start

Making a traceTorch model is barely any different from PyTorch models. Here's how:

### 1. The "zero-boilerplate" module

Inherit from `tracetorch.snn.TTModule` instead of `pytorch.nn.Module`. This gives your model the powerful recursive
methods like `zero_states()` and `detach_states()` for free, while still integrating with other PyTorch `nn.Module`.

```python
import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn


class ConvSNN(snn.TTModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            # dim=-3 tells the layer that the 3rd-to-last dimension is the channel dim.
            # This works for (B, C, H, W) AND unbatched (C, H, W) inputs automatically.
            snn.LIF(num_neurons=32, beta=0.9, dim=-3),

            nn.Flatten(),
            nn.Linear(32 * 26 * 26, 128),

            # Readout layer with learnable decay initialized to scrape various timescales
            snn.Readout(128, beta=torch.rand(128)),
            # Map the readout layer back down to the desired number of dimensions
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)
```

### 2. The Training Loop

State management is easily handled outside the forward pass. Simply call `.zero_states()` on the model to reset all
hidden states to `None`, and call `.detach_states()` to detach the current hidden states (used in truncated BPTT or for
online learning).

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConvSNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training Step
model.train()
for x, y in dataloader:
    x, y = x.to(device), y.to(device)

    model.zero_states()  # Crucial: Reset hidden states for the batch
    optimizer.zero_grad()

    # Time loop
    spikes = []
    for step in range(num_timesteps):
        # Just pass x. No state tuples to manage.
        spikes.append(model(x))

    # Stack output and compute loss
    output = torch.stack(spikes)
    loss = loss_fn(output.mean(0), y)  # Rate coding example

    loss.backward()
    optimizer.step()
```

## Documentation

The online documentation can be found [here](https://yegor-men.github.io/tracetorch/). It contains the theory behind
SNNs, the traceTorch API and layers available, as well as a couple tutorials to recreate the code found in
`examples/`.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to fork, submit pull requests or report issues, I will occasionally check in
on it.
