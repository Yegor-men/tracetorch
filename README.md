![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/license/mit)
[![PyPI](https://img.shields.io/badge/PyPI-v0.9.2-blue.svg)](https://pypi.org/project/tracetorch/)
[![Documentation](https://img.shields.io/badge/Documentation-v0.9.2-green.svg)](https://yegor-men.github.io/tracetorch/)

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

## Documentation

The online documentation can be found [here](https://yegor-men.github.io/tracetorch/). It is thoroughly recommended to
at least read the introduction section as it contains the theory behind SNNs, the traceTorch API and layers available,
as well as a couple tutorials to recreate the code found in `examples/`.

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

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to fork, submit pull requests or report issues, I will occasionally check in
on it.

## Roadmap

traceTorch still has a long way to go. Namely, in no particular order:

- Finish `examples/` section for code
- Create simple tests to assert working order
- Finish `introduction/` section of the docs
- Do the `reference/` section for the docs
- Do the `tutorials/` section for the docs, basing it on the `examples/`
- Make docstrings
- Make pos/neg split for `syn` and `rec`, make it optional, with separate decays
- Figure out what stuff matters for non `snn/` section