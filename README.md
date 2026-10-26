![traceTorch Banner](https://raw.githubusercontent.com/Yegor-men/tracetorch/main/media/tracetorch_banner.png)

[![Documentation](https://img.shields.io/badge/Documentation-online-blue)](https://yegor-men.github.io/tracetorch/)
[![PyPI version](https://img.shields.io/pypi/v/tracetorch)](https://pypi.org/project/tracetorch/)
[![License](https://img.shields.io/badge/License-MIT-purple)](https://opensource.org/license/mit)

# traceTorch

An image can arrive one patch at a time. Audio arrives over time, and the next byte in a passage depends on what came before. In each case, a model needs a way to retain information while processing new input.

traceTorch brings that memory to ordinary PyTorch models, with a focus on spiking neural networks. A layer processes one timestep and keeps its own hidden states, so you can place recurrent neurons between familiar linear layers and write the sequence loop yourself:

```python
import torch
from torch import nn
import tracetorch as tt

class Model(tt.Model):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 64),
            tt.snn.LIB(64),
            nn.Linear(64, 10),
            tt.snn.LI(10, beta=0.9, beta_rank=0, learn_beta=False),
        )

    def forward(self, x):
        return self.layers(x)

model = Model()
sequence = torch.randn(49, 8, 16)  # 49 timesteps, 8 independent sequences
model.reset_states()
for timestep in sequence:
    scores = model(timestep)
```

The last scores contain the result of processing the sequence. You choose whether to train from those final scores or from every timestep, when to begin a new sequence, and how long gradients should follow its history. Parameters work with PyTorch optimizers and checkpoints; memory can be reset, detached, saved and restored across the whole model.

That makes traceTorch a place to build and understand recurrent models without hiding the update behind a training framework. You can start with a built-in neuron, inspect its equations, or define a new layer whose states have the shapes your idea needs—including matrices. The [documentation](https://yegor-men.github.io/tracetorch/) connects these operations to runnable models and illustrated notebooks.

## Installation

```bash
pip install tracetorch
```

PyTorch is the library's runtime dependency. The examples list their own dataset, plotting and notebook packages, so you can install those when you choose an example to work through.

## Quickstart

We'll start with a small model that remembers a random value through seven timesteps of silence, then reproduces it from the final output. It needs no dataset download, and the complete training loop fits in one file:

```bash
git clone https://github.com/Yegor-men/tracetorch.git
cd tracetorch
pip install -e .
python examples/quickstart/main.py
```

The [quickstart explanation](https://yegor-men.github.io/tracetorch/quickstart.html) follows the [program](examples/quickstart/main.py) through model construction, state reset, the timestep loop and a weight update. [Understanding traceTorch](https://yegor-men.github.io/tracetorch/introduction/index.html) then connects those choices to the layer equations and the library's design.

## Examples and tutorials

Each example has its own folder with `main.py`, a local `requirements.txt`, and a `tutorial.ipynb` when a walkthrough is available. The program runs the complete workflow, while the notebook builds the same solution with explanations, figures and experiments. The documentation displays that notebook directly, so the lesson stays the same on the website and in your editor.

For MNIST, install the local requirements and run the complete program:

```bash
pip install -r examples/mnist/requirements.txt
python examples/mnist/main.py --device cpu
```

To work through its notebook instead, run `jupyter lab` and open `examples/mnist/tutorial.ipynb`. You can also use an IDE with Jupyter support in the same Python environment.

| Example | Notebook | Complete program |
| --- | --- | --- |
| Reading image patches with an SNN | [tutorial.ipynb](examples/mnist/tutorial.ipynb) | [main.py](examples/mnist/main.py) |
| Retaining auditory features with synaptic residual blocks | [tutorial.ipynb](examples/shd/tutorial.ipynb) | [main.py](examples/shd/main.py) |
| Predicting bytes with a residual SNN and online updates | [tutorial.ipynb](examples/alice_language_model/tutorial.ipynb) | [main.py](examples/alice_language_model/main.py) |
| Building a neuron with matrix-shaped histories | [tutorial.ipynb](examples/custom_neuron/tutorial.ipynb) | [main.py](examples/custom_neuron/main.py) |

These are teaching experiments: they demonstrate model construction, training and interpretation with small workloads, rather than aiming for benchmark-leading results. All examples reuse `examples/data/`, while plots and checkpoints belong to each example's folder. The defaults keep training short on CPU; an initial dataset download can take longer. [examples/README.md](examples/README.md) describes the commands and saved files.

## Available layers

The library includes 32 leaky-integrator variants with continuous or firing outputs, plus SimpleRNN, GRU and LSTM layers. You can choose deterministic, stochastic or smooth firing, add synaptic and recurrent histories, or use `tt.Layer` to define a different state update.

Built-in recurrent layers use the final dimension for features. `tt.Model` coordinates their state operations, while parameter compilation can cache transformed values during evaluation and decompilation restores their intended learnability. The [API reference](https://yegor-men.github.io/tracetorch/reference/index.html) gives the exact signatures and equations.

## Author and license

Created by [Yegor Menovchshikov](https://github.com/Yegor-men) and distributed under the MIT license.
