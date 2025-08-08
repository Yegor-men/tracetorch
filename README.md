![traceTorch Banner](media/tracetorch_banner.png)

# traceTorch

[![License](https://img.shields.io/badge/License-Apache%202.0-purple.svg)](https://www.apache.org/licenses/LICENSE-2.0)

A memory efficient, autograd-graph-free PyTorch extension for training recurrent Spiking Neural Network (SNN) models
online, using input traces for constant‐memory backpropagation.

## 🚀 Key Features

- **Graph‑Free Backpropagation**: Execute backward passes without ever building or storing autograd graphs, the memory
  usage remains constant, even over arbitrarily long sequences.
- **Trace‑Driven Gradients**: Each layer maintains a learnable exponential input trace that compactly encodes temporal
  history, allowing local reconstruction of average inputs and outputs for fully local gradient estimation.
- **Intermittent, Online Learning**: Invoke `.backward(ls)` at any timestep cadence; per update memory and compute cost
  stays constant, making it ideal for streaming and continual learning.
- **Spiking Neural Network Architecture**: Runs on standard LIF neurons, stochastic softmax classifiers and noisy
  Bernoulli‑encoded inputs for biologically inspired learning.

## Roadmap

- Implement the trace alternative to REINFORCE
- Write docstrings
- Write documentation
- Make traceTorch into a PyPI library

## Installation

⚠️ WARNING, traceTorch is _not yet_ a library. For now you'll just have to clone this repository:

```
git clone https://github.com/Yegor-men/tracetorch
```

Then, within a python file where from where the `tracetorch/` folder is visible, simply do:

```
import tracetorch
```

## Usage/Examples

There exists `tracetorch/tests/` within which sit test files for playtesting, aimed to test if the components work. So
far:

- `lif_test.py` tests if a model comprised of LIF layers can memorize the mapping of some random spike input to some
  random spike output. The loss and ls are constructed only from the last timestep. Knowing that the theoretically most
  maximally compressed latent should be indistinguishable from noise, this is a good confirmation that the model can
  perform calculations and computations over time to get some desired output.
- `classification_test.py` is the same as `lif_test.py` with the exception of the last layer now being a classification
  one, the idea is to see if the model can memorize the mapping of some random input to a classification. Knowing that
  the theoretically most maximally compressed latent should be indistinguishable from noise, this is a good confirmaiton
  that the model can work to be a "decoder" of sorts.
- `mnist.py` tests if a traceTorch model can learn on MNIST, hence if it can generalize to unseen data. Also is added
  rate coding by first transforming all the images to be compressed to some range, and then passing them through
  `torch.bernoulli()`, hence creating noisy input over time, thus further testing the ability to generalize.

## How Does traceTorch Work?

### 1. Overview

traceTorch replaces PyTorch’s autograd graph with a compact, per-layer “input trace” that accumulates past inputs into a
single tensor. When it’s time to learn, traceTorch inverts those traces to recover average inputs and outputs, then
computes gradients locally—no heavyweight graph ever grows.

### 2. Maintaining an Input Trace

Each layer keeps one tensor, `trace`, which is updated every timestep:

```python
trace = decay * trace + input_spikes
````

* **decay** ∈ (0, 1) is learnable
* **input\_spikes** are the binary or real-valued activations at that step

Over time, `trace` exactly represents the exponentially weighted sum of all past inputs.

### 3. Reconstructing Average Inputs

When you call `.backward()`, you don’t have to replay every timestep. Assuming the trace has stabilized, you can solve:

```text
trace = decay * trace + input
⇒ input = trace * (1 – decay)
```

This recovers the **average input** over the recorded interval without storing any intermediate tensors.

### 4. Estimating Average Outputs

Using the reconstructed average input, each layer estimates its **average firing rate** (or activation) analytically.
This smooth approximation stands in for the full spike train or activation history.

### 5. Local, Graph-Free Backpropagation

Armed with average inputs and outputs, each layer computes:

* Parameter gradients (e.g. decay, membrane leak, thresholds)
* Upstream learning signals (derivative of loss w\.r.t. its inputs)

All updates happen **locally**, layer by layer. Since no autograd graph was built, memory stays constant—even across
thousands of timesteps.

### 6. Why It Works

* **Weighted history**: Exponential decay naturally emphasizes recent events, matching biological intuition and temporal
  credit assignment.
* **Mathematical equivalence**: The running trace exactly encodes the same information as storing every input and
  weighting it later.
* **Constant memory**: One tensor per layer replaces potentially unbounded sequence histories.

This approach yields the same end-to-end gradient information as BPTT, but without ever constructing or traversing a
giant computation graph.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Acknowledgements

I built tracetoch from the ground up, trying to reverse engineer biological neurons with a sprinkle of intelligent
design, but I would also like to recognize the following projects and people who helped shape my thinking:

- [snntorch](https://github.com/jeshraghian/snntorch) for introducing me to SNN networks in the first place, and their
  principles of function (albeit it works to be integrated with base pytorch, hence building the autograd graph during
  `.forward()`).
- [Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov) for introducing me to computational neuroscience, presenting
  interesting concepts in an easy to understand manner. My earliest tests, when I naively wanted to implement 1:1
  biological neurons, largely revolved around his work.
- [e-prop (eligibility propagation)](https://www.biorxiv.org/content/10.1101/738385v4) inspired the whole "trace"
  concept, the idea of keeping a decaying value. Earlier, before traceTorch, I wanted to use e-prop for online learning
  instead. Admittedly unsuccessful in my attempts, and a little put off by the relative difficulty, I instead wanted to
  make something simpler.

## Contributing

Contributions are always welcome. Feel free to fork the repo or submit issues, I will occasionally check in on it.

You can also reach out to me via either email or Twitter:

- email: yegor.mn@gmail.com
- [Twitter](https://x.com/Yegor_Men)
