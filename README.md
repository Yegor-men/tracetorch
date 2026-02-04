![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/license/mit)
[![PyPI](https://img.shields.io/badge/PyPI-v0.6.0-blue.svg)](https://pypi.org/project/tracetorch/)

# traceTorch

A strict, ergonomic, and powerful Spiking Neural Network (SNN) library for PyTorch.

traceTorch is bult around a single, highly compositional neuron superclass, replacing the restrictive "layer zoo" of
countless disjoint neuron types with the `LeakyIntegrator`. This design encapsulates a massive range of SNN dynamics:

- synaptic and recurrent filtering
- rank-based parameter scoping for scalar, per-neuron or matrix weights
- optional Exponential Moving Average (EMA) on any hidden state
- arbitrary recurrence routing to any hidden state
- flexible polarity for spike outputs: positive and/or negative

All into declarative configuration on one class. By abstracting this complexity, traceTorch provides both the robust
simplicity required for fast prototyping via familiar wrappers (`LIF`, `RLIF`, `SLIF`, `Readout`, etc.) and the
unprecedented flexibility required for real research.

## Why traceTorch?

Existing SNN libraries often feel restrictive or require verbose state management. Aside from the technical features and
capabilities, traceTorch follows a different philosophy, revolving around ergonomics:

- **Architectural Flexibility:** All existing traceTorch layers are just small wrappers of the `LeakyIntegrator`
  superclass, and it's incredibly easy to add your own alterations/combinations of the features you like.
- **Automatic State Management:** No need to manually pass hidden states through `.forward()`, each layer manages its
  own hidden states, and calling `.zero_states()` on a traceTorch model recursively clears _all_ the hidden states the
  entire model uses, no matter how deeply hidden they are. In a similar style, `.detach_states()` detaches the states
  from the current computation graph.
- **Lazy Initialization:** Hidden states are initialized as `None` and allocated dynamically based on the input shape.
  This completely eliminates "Batch Size Mismatch" errors during inference.
- **Dimension Agnostic:** Whether you are working with `[Time, Batch, Features]` or `[Batch, Channels, Height, Width]`
  tensors, layers _just_ work. Change a single `dim` argument during layer initialization to indicate the target
  dimension the layer acts on. Defaults to `-1` for MLP, `-3` would work for CNN (channels are 3rd last in
  `[B, C, H, W]` or `[C, H, W]`).
- **Smooth Constraints:** Parameters like decays and thresholds are constrained via Sigmoid and Softplus respectively.
  No hard clamping, meaning that gradients flow smoothly and accurately everywhere.
- **Rank Based Parameters:** Instead of messy flags like `*_is_vector` or `all_to_all`, traceTorch uses a single
  `*_rank` integer to define the parameter scope: 0 for a scalar (parameter is shared across the layer), 1 for a
  vector (per-neuron parameter), 2 for a matrix (dense all-to-all connections for recurrent layer weights).

## Installation

traceTorch is a PyPI library found [here](https://pypi.org/project/tracetorch/). Requirements are listed in
`requirements.txt`.

```bash
pip install tracetorch
```

## Quick Start

Making a traceTorch model is barely any different from PyTorch models. Here's how:

### 1. The "zero-boilerplate" module

Inherit from `tracetorch.snn.TTModule` instead of `pytorch.nn.Module`. This gives your model powerful recursive methods
like `zero_states()` and `detach_states()` for free, while still integrating with other PyTorch `nn.Module`.

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
			nn.Linear(32 * 26 * 26, 10),

			# Readout layer with learnable scalar decay
			snn.Readout(num_neurons=10, beta=0.8, beta_rank=0)
		)

	def forward(self, x):
		return self.net(x)
```

### 2. The Training Loop

State management is easily handled outside the forward pass. Simply call `.zero_states()` on the model to reset all
hidden states to `None`, or call `.detach_states()` to detach the current hidden states (used in truncated BPTT or for
online learning).

```python
model = ConvSNN().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
loss_fn = tt.loss.soft_cross_entropy  # Handles non-onehot targets gracefully

# Training Step
for x, y in loader:
	x, y = x.cuda(), y.cuda()
	model.train()

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
	optimizer.zero_grad()

	# Crucial: Reset hidden states for the next batch
	model.zero_states() 
```

## Documentation

The online documentation can be found [here](https://yegor-men.github.io/tracetorch/). It contains introductory lessons
to SNNs, the traceTorch API and layers available, as well as a couple tutorials to recreate the code found in
`examples/`.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to fork, submit pull requests or report issues, I will occasionally check in
on it.
