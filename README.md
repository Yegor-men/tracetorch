![traceTorch Banner](media/tracetorch_banner.png)

[![Documentation](https://img.shields.io/badge/Documentation-v0.15.0-red.svg)](https://yegor-men.github.io/tracetorch/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/license/mit)
[![PyPI](https://img.shields.io/badge/PyPI-v0.15.0-blue.svg)](https://pypi.org/project/tracetorch/)

# traceTorch

A strict, ergonomic, and powerful Spiking Neural Network (SNN) library for PyTorch.

## Introduction

traceTorch is an SNN library written from the ground up with power, flexibility and extensibility in mind. As with any
other library, traceTorch presents a wide and powerful variety of distinct, commonly used neuron types that utilize
sensible defaults. The naming schema and API, albeit a bit unconventional, are consistent and self-explanatory, and
allow to control a massive variety of options. When necessary, parameters are bound by activation functions so that the
gradients flow smoothly and are never clamped. A single dimension argument during initialization determines which of the
tensor's dimensions the layer is looking at, so that the same layer can work with various tensor shapes.

traceTorch also helps with management of hidden states, no matter how deeply they are buried in the model. They are
lazily initialized, and inheriting from the model superclass unlocks helper methods to find and manage them en-masse:
saving and loading, zeroing and detaching. The model superclass also allows you to "compile" and "uncompile" the model
so that for inference, the necessary parameters don't need to be passed through their activation function each time.

If the existing layers aren't enough, traceTorch also helps with creating your own SNN layers that comply with the rest
of the traceTorch API. Inheriting from the layer superclass unlocks helper methods to initialize parameters:
learnability, rank, inverse functions (if applicable).

All in all, traceTorch exists to make writing, reading, debugging, and most importantly: experimenting, with SNNs in
PyTorch to feel significantly more natural and less frustrating than in existing alternatives, while preserving (and in
many cases enhancing) the expressive power needed for real models and research. traceTorch ultimately rewards users who
value minimalism, composition, and long-term extensibility.

## Features

traceTorch follows a slightly unconventional, but consistent and self-explanatory naming schema. The names are modular
and explain their role and function.

- `LI` base name stands for `Leaky Integrator`: the simplest of layer types with just one trace and decay: the membrane
  potential and the beta decay. No firing and no reset mechanics, this layer type is commonly known as `Readout` (
  although it's not recommended to literally have it as the final layer).
- `~B` suffix stands for `Binary`, the presence of a strictly positive threshold, meaning that the layer has 2 possible
  outputs: a 1 or a 0. `LIB` is hence the official name for the `LIF`.
- `~T` suffix stands for `Ternary`, meaning that the layer has 2 thresholds: a strictly positive and a strictly negative
  one, meaning that the layer has 3 possible outputs: 1, 0 or -1.
- `~S` suffix is only used with the `~T` suffix to create `~TS`, which stands for `Ternary Scaled`, meaning that the
  ternary outputs are multiplicatively separately scaled based on their polarity. This is done so that the three
  possible outputs are truly independent when we consider the downstream layer.
- `D~` prefix stads for `Dual`, meaning that all traces (hidden states) and their decay parameters are split into a
  separate positive and negative version for greater expressivity and unlocking more complex dynamics.
- `S~` prefix stands for `Synaptic`, meaning that before the membrane there is a separate synaptic trace with its
  respective alpha decay that smooth out the inputs over time via an exponential moving average (EMA) before they get
  integrated into the membrane.
- `R~` prefix stands for `Recurrent`, meaning that the layer records its own outputs into a separate trace with its own
  gamma decay and re-integrates it back into the membrane in the next timestep. The computation graph is made to work
  even with online learning.

In total, this results in 28 specially made, performant layers which easily integrate and work with other PyTorch
layers: `LI`, `DLI`, `SLI`, `DSLI`, `LIB`, `DLIB`, `SLIB`, `RLIB`, `DSLIB`, `DRLIB`, `SRLIB`, `DSRLIB`, `LIT`, `DLIT`,
`SLIT`, `RLIT`, `DSLIT`, `DRLIT`, `SRLIT`, `DSRLIT`, `LITS`, `DLITS`, `SLITS`, `RLITS`, `DSLITS`, `DRLITS`, `SRLITS`,
`DSRLITS`.

However, layers also have a plethora of extra features:

- Rank-based parameter scoping for per-layer (scalar) or per-neuron (vector) parameters, defaulting to per-neuron.
- Initialize parameters via a float value or your own desired tensor.
- Make any parameter learnable or static, automatically set to an `nn.Parameter` or registered buffer accordingly.
- Single `dim=` argument determines the target dimension the layer focuses on: `-1` for MLP, `-3` for CNN, et cetera.
- Smooth parameter constraints for those that require it (sigmoid on decays and softplus on thresholds), meaning that
  gradients always flow cleanly and accurately. The respective inverse function is applied if necessary during
  initialization.
- All the layers with reset mechanics (`~LIB`, `~LIT`, `~LITS`) also have a `spike_fn` and `quant_fn`. The former is
  used to turn the membrane into a "probability" to fire, and the latter actually turns that probability into the
  output. `spike_fn` defaults to sigmoid(4x) because of the nice constraints and gradients, and `quant_fn` presents 3
  options: `round`, `bernoulli`, `probabilistic`; which round, take a differentiable bernoulli sample, and take a
  differentiable bernoulli sample multiplied by the probability respectively, but defaults to `bernoulli` for stability.

traceTorch also presents the `TTModel` superclass, which is used for model managing. Inheriting from the `TTModel` class
to grants access to recursive methods `.zero_states()` and `.detach_states()` to recursively respectively set the states
to `None` or to detach; and `.save_states()` and `.load_states()` to save and load hidden states, working both with
`.pt` and `.safetensors`, no matter how deeply hidden they are: PyTorch modules such as `nn.Sequential` or python
classes and data structures; it doesn't matter. There is also `.TTcompile()` and `TTuncompile()` to compile and
uncompile a tracetorch model: so that the decays and thresholds are saved as-is and don't get passed through the
activation function in each timestep.

The `TTLayer` superclass handles all the boilerplate of creating SNN layers. Instead of wrestling with parameter
registration, state management, and dimension handling, you can just inherit from `TTLayer` to create your own SNN
layers that comply with the traceTorch ethos. It handles:

- Automatic parameter registration: rank, learnability, value / tensor initialization and inverse functions
  for decays and thresholds.
- State management for hidden states: methods to bulk zero / detach / initialize hidden states for the layer (
  `TTModel` is for working with the layers in a model, `TTLayer` is for managing the states in the layer itself).
- Dimension helpers: methods to move a tensor's dimension (the `dim=` used during initialization) to the last
  dimension so that the layer is tensor shape agnostic.
- Property generation: parameters are saved in `raw_*` form to account for inverse and activation functions, but
  work intuitively such that `layer.beta` returns the sigmoid activated value, et cetera.
- Compiling and uncompiling a model: `TTcompile` and `TTuncompile` to get rid and respectively re-add the `raw_*`
  parameters and activation functions and just use the values directly.

## Documentation

The online documentation can be found [here](https://yegor-men.github.io/tracetorch/). It is thoroughly recommended to
at least read the introduction section before proceeding as it contains the theory behind SNNs, the traceTorch ethos and
layers available as well as a brief explanation of what it is that each mechanic actually does. It also contains a
couple tutorials to recreate the code found in `examples/`.

## Installation

traceTorch is a PyPI library found [here](https://pypi.org/project/tracetorch/). Requirements for the library are listed
in `requirements.txt`. Take note that examples found in `examples/` may have their own requirements, separate from the
library requirements.

```bash
pip install tracetorch
```

If you don't want to install traceTorch as a library, or just want to test the examples, you should install traceTorch
as an editable installation:

```bash
git clone --branch v0.15.0 https://github.com/Yegor-men/tracetorch
cd tracetorch
pip install -e .
```

Make sure to check the [releases](https://github.com/Yegor-men/tracetorch/releases) page for the latest (or different)
version number if you want a different release.

## Quickstart

traceTorch models look barely any different from PyTorch models:

```python
import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

device = "cuda" if torch.cuda.is_available() else "cpu"


class SNN(snn.TTModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            snn.LIB(16, dim=-3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            snn.LIB(64, dim=-3),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            snn.LI(128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


model = SNN().to(device)
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)

# TRAINING LOOP WITH DATALOADER
model.train()
for x, y in train_dataloader:
    model.zero_states()  # sets hidden states to None for lazy assignment
    model.zero_grad()
    running_loss = 0.0
    for t in range(num_timesteps):
        model_output = model(x[t])
        loss = loss_fn(model_output, y[t])
        running_loss = running_loss + loss
        # optionally call model.detach_states() for online learning here
    running_loss.backward()
    optimizer.step()
```

## Examples

Example code can be found in `examples/`. To test the code, make sure that you have the respective requirements
installed for the example, and that you've either installed traceTorch from PyPI or as an editable installation.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to fork, submit pull requests or report issues, I will occasionally check in
on it.

## Roadmap

traceTorch still has a long way to go. Namely:

- Clean up the experimental `TTcompile`, `TTuncompile`, `save_states`, `load_states` methods for `TTModel`
- Fix the `LeakyIntegrator` superclass and create the 28 tests
- Finish the `examples/` section for example code for various examples
- Make proper requirements for each example in `examples/`
- Finish the `introduction/` section of the docs
- Do the `reference/` section for the docs
- Do the `tutorials/` section for the docs, basing it on the `examples/`
- Make docstrings
- Figure out versioning requirements for the library