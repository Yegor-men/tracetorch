![traceTorch Banner](https://raw.githubusercontent.com/Yegor-men/tracetorch/main/media/tracetorch_banner.png)

[![Documentation](https://img.shields.io/pypi/v/tracetorch?style=flat&labelColor=555&label=Documentation&color=red)](https://yegor-men.github.io/tracetorch/)
[![PyPI version](https://img.shields.io/pypi/v/tracetorch?style=flat&labelColor=555&label=PyPI&color=blue)](https://pypi.org/project/tracetorch/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat&labelColor=555)](https://opensource.org/license/mit)
[![GitHub issues](https://img.shields.io/github/issues/Yegor-men/tracetorch?style=flat&labelColor=555&label=Issues&color=orange)](https://github.com/Yegor-men/tracetorch/issues)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tracetorch?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=Downloads)](https://pepy.tech/projects/tracetorch)

# traceTorch

traceTorch is a PyTorch library for stateful recurrent layers, built primarily for spiking neural networks.

It gives you SNN, RNN, and SSM-style layers that behave like ordinary PyTorch modules: one tensor in, one tensor out, hidden states kept inside the layer. The difference is that those hidden states are still easy to manage. Inherit from `tt.Model`, call `zero_states()`, `detach_states()`, `save_states()`, `load_states()`, `TTcompile()`, or `TTdecompile()`, and traceTorch handles every traceTorch layer buried inside the model.

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
            tt.snn.LIB(num_neurons=128),
            nn.Linear(128, 10),
            tt.snn.LI(num_neurons=10),
        )

    def forward(self, x):
        return self.net(x)


model = Net()
model.reset_states()
out = model(torch.rand(32, 1, 28, 28))
```

## Why traceTorch?

- **Hidden states stay hidden.** Layers own their states, so model code stays readable.
- **State management is explicit.** Reset between sequences with `zero_states()`, truncate history with `detach_states()`, and save/load hidden states when needed.
- **SNNs are first-class.** `tt.snn` contains 32 leaky-integrator-based layers with binary, ternary, scaled ternary, continuous, dual, synaptic, and recurrent variants.
- **PyTorch composition stays normal.** Put traceTorch layers inside `nn.Sequential`, CNNs, MLPs, and custom PyTorch modules.
- **Feature dimensions are configurable.** Use `dim=-1` for MLP features, `dim=-3` for image channels, or any other target dimension.
- **Parameters can be scalar, per-neuron, fixed, learnable, or tensor-initialized.**

## Layer Families

| Module | Layers |
| --- | --- |
| `tt.snn` | `LI`, `LIB`, `LIT`, `LITS` families, including dual (`D`), synaptic (`S`), recurrent (`R`), and combined variants |
| `tt.rnn` | `SimpleRNN`, `LSTM`, `GRU` |
| `tt.ssm` | `S4`, `S5`, `S6`, `Mamba` adapted to traceTorch's one-timestep recurrent interface |

traceTorch's main focus is SNN experimentation. The RNN and SSM layers exist because the same state-management design is useful there too, but the SSM implementations are not meant to replace the official optimized sequence-parallel versions.

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
