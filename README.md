![traceTorch Banner](https://raw.githubusercontent.com/Yegor-men/tracetorch/main/media/tracetorch_banner.png)

[![Documentation](https://img.shields.io/pypi/v/tracetorch?style=flat&labelColor=555&label=Documentation&color=red)](https://yegor-men.github.io/tracetorch/)
[![PyPI version](https://img.shields.io/pypi/v/tracetorch?style=flat&labelColor=555&label=PyPI&color=blue)](https://pypi.org/project/tracetorch/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat&labelColor=555)](https://opensource.org/license/mit)
[![GitHub issues](https://img.shields.io/github/issues/Yegor-men/tracetorch?style=flat&labelColor=555&label=Issues&color=orange)](https://github.com/Yegor-men/tracetorch/issues)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tracetorch?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/tracetorch)

# traceTorch

traceTorch is a PyTorch library for stateful recurrent layers, built primarily for spiking neural networks.

It provides tools for building stateful layers, with SNN and RNN layers bundled by default. Hidden states stay inside each layer and are easy to manage: inherit from `tt.Model` and use `reset_states()`, `detach_states()`, `save_states()`, `load_states()`, `compile_parameters()`, or `decompile_parameters()` across the model.

```bash
pip install tracetorch
```

```python
import torch
from torch import nn
import tracetorch as tt


class Net(tt.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            tt.snn.LIB(num_features=128),
            nn.Linear(128, 10),
            tt.snn.LI(num_features=10),
        )

    def forward(self, x):
        return self.net(x)


model = Net()
model.reset_states()
out = model(torch.rand(32, 1, 28, 28))
```

## Why traceTorch?

- **Hidden states stay hidden.** Layers own their states, so model code stays readable.
- **State management is explicit.** Reset between sequences with `reset_states()`, truncate history with `detach_states()`, and save/load hidden states when needed.
- **SNNs are first-class.** `tt.snn` contains 32 leaky-integrator-based layers with binary, ternary, scaled ternary, continuous, dual, synaptic, and recurrent variants.
- **PyTorch composition stays normal.** Put traceTorch layers inside `nn.Sequential`, CNNs, MLPs, and custom PyTorch modules.
- **Built-in layers use features last.** Move image channels explicitly with `tt.utils.MoveDim` when composing with convolutions.
- **Parameters can be scalar, per-neuron, fixed, learnable, or tensor-initialized.**

## Layer Families

| Module | Layers |
| --- | --- |
| `tt.snn` | `LI`, `LIB`, `LIT`, `LITS` families, including dual (`D`), synaptic (`S`), recurrent (`R`), and combined variants |
| `tt.rnn` | `SimpleRNN`, `LSTM`, `GRU` |

traceTorch's main focus is SNN experimentation. The base layer can also support custom vector or matrix states, including state-space dynamics.

## Custom states and parameters

`tt.Layer()` has no global feature count or target dimension. Declare each state with its own trailing shape:

```python
self.define_state("mem", (512,))
self.define_state("history", (512, 4), dim=-1)
```

`zero_states(x)` lazily creates zeros with shape `x.shape[:dim] + state_shape`, using `x`'s dtype and device. Negative `dim` removes trailing input dimensions; `None` removes none. Existing states are preserved until reset. Time iteration remains external to the layer.

`define_parameter(name, value, learnable=True, initialization_fn=None, activation_fn=None)` accepts a tensor of any shape. Initialization transforms it into an unconstrained raw `nn.Parameter` once; activation transforms it when accessed. SNN helpers retain scalar/per-neuron convenience arguments. Use `set_parameter_learnable(name, bool)` to change intended learnability.

Compilation caches detached activated values in nonpersistent buffers and temporarily freezes raw parameters. Decompilation removes the caches and restores intended learnability without replacing parameters or invalidating optimizers. Loading weights clears caches too.

Save weights with `torch.save(model.state_dict(), path)` and hidden states separately with `torch.save(model.save_states(), path)`. Neither checkpoint contains the other's tensors, and compiled caches are never included in weight state dictionaries. Learnability settings are configuration, not weight tensors. Move the model to its target device/dtype before allocating or loading plain-tensor hidden states.

For channel-first images:

```python
nn.Sequential(
    nn.Conv2d(3, 32, 3),
    tt.utils.MoveDim(-3, -1),
    tt.snn.LIB(32),
    tt.utils.MoveDim(-1, -3),
)
```

## Documentation

Read the full documentation at <https://yegor-men.github.io/tracetorch/>.

Recommended path:

1. **Installation**: install the package or editable repository.
2. **Quickstart**: build and train a minimal traceTorch model.
3. **Introduction**: understand the state model, SNN naming scheme, and design choices.
4. **Examples**: follow MNIST and Heidelberg Digits examples from `examples/`.
5. **Tutorials**: learn saving/loading states, compiling/decompiling, and custom layer creation.
6. **Reference**: inspect API docstrings.

## Examples

Runnable examples live in `examples/`.

```bash
git clone https://github.com/Yegor-men/tracetorch.git
cd tracetorch
pip install -e .

cd examples/mnist
pip install -r requirements.txt
python rate_coded.py
```

Current examples:

- `examples/mnist/rate_coded.py`: rate-coded MNIST over repeated Bernoulli timesteps.
- `examples/mnist/sequential.py`: MNIST as a patch sequence.
- `examples/mnist/noisy.py`: MNIST with noisy repeated observations.
- `examples/heidelberg_digits/main.py`: Spiking Heidelberg Digits with Tonic.

These examples are written to show traceTorch mechanics clearly. They are not tuned as benchmark or SOTA training recipes.

## Development Status

traceTorch is approaching its v1.0.0 release. The current focus is documentation, tests, examples, and API polish.

## Author

Created by [Yegor Menovchshikov](https://github.com/Yegor-men).

## License

MIT.
