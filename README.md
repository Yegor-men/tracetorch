![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/license/mit)
[![PyPI](https://img.shields.io/badge/PyPI-v0.3.0-blue.svg)](https://pypi.org/project/tracetorch/)

# traceTorch

A small, lightweight, and highly opinionated spiking neural network library for PyTorch.

`traceTorch` is a from-scratch, clean-room reimplementation of classic spiking neuron dynamics, integrating with the
PyTorch autograd engine. `traceTorch` is heavily inspired by [snnTorch](https://github.com/jeshraghian/snntorch) (
MIT-licensed); many class names (`leaky`, `synaptic`), variable names (`mem`, `syn`), and general API patterns are
intentionally kept similar because they feel like natural and sensible choices, alterations would add needless
complexity.

Key differences and personal preferences baked in:

- Parameters such as decays and thresholds are intentionally constrained with sigmoid and softplus for conceptual
  cleanliness
- Parameters are initialized as either nn.Parameter or as a registered buffer for if they should be learnable or not
- Parameters can toggle between scalar and vector with zero code branching: they are internally decomposed into a scalar
  and vector. The scalar is initialized to the desired default value and the vector is initialized to ones. Making the
  scalar learnable and the vector not makes the parameter act like a scalar, setting both to learnable makes it act like
  a vector. @property methods are added so that you can just get the parameter by the name, what's returned is the
  effective parameter value used in the forward pass (`activation_fn(param_scalar * param_vector)`)
- Some handy utilities that I like to use, such as plotting or recording of the parameters and model states
- Much smaller codebase with essentially zero features I don’t use myself

Important notes

- `traceTorch` doesn't (intentionally) share source code from `snnTorch` or any other library, no code was copied,
  although similarities are most certainly very likely
- `traceTorch` is primarily written for my own daily use. Documentation is minimal, tutorials effectively don’t exist,
  and some parts are vibecoded rather than cleanly written
- If you want a mature, well-documented, SNN library, it's recommended to use `snnTorch` or any of the other SNN
  libraries. They are likely to be better

You are of course welcome to use `traceTorch` if the design choices align with your requirements or preferences, but
there are no guarantees of stability, completeness, cleanliness or documentation. Think of `traceTorch` as "yet another
personal take on an already existing wheel", rather than a competitor.

## Installation

`traceTorch` is a PyPI library, which can be found [here](https://pypi.org/project/tracetorch/).

You can install it via pip. Requirements for the library are listed in `requirements.txt`.

```
pip install tracetorch
```

To use, it is recommended to import as such:

```
import tracetorch as tt
from tracetorch import snn
```

## Usage examples

`examples/` contains various PyTorch models and the associated training code that make use traceTorch. The examples make
use of libraries not listed in the requirements, but aren't anything fancy or rare. The folder is primarily used for my
own projects and experiments, completeness or cleanliness is not guaranteed, although they are numbered for the sake of
simplicity.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Contributing

Contributions are always welcome. Feel free to fork, submit pull requests or report issues, I will occasionally check in
on it.
