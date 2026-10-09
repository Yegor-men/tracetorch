# Test suite

Install the test dependencies and run from the repository root:

```sh
pip install -e '.[test]'
python -m pytest
```

Tests are organized by permanent behavior, not by development milestones:

- `core/test_layer.py`: parameter/state definitions, lazy allocation, reset/detach, learnability, compilation, and weight loading. These tests use the generic base layer.
- `core/test_model.py`: recursive operations, containers/shared objects, model overrides, and independent hidden-state checkpoints. Minimal test layers isolate the model machinery from neuron dynamics.
- `test_snn.py`: all 32 public variants against an independent scalar oracle, including smooth firing; hand-calculated trajectories, threshold boundaries, delayed reset, separate signed event history, output-scale isolation, and independent parallel instances.
- `test_rnn.py`: trajectories against mapped PyTorch RNN/LSTM cells and explicit scalar GRU equations, plus shape and instance-independence checks.
- `test_autograd.py`: analytical temporal derivatives, detach/reset boundaries, numerical gradient checks for every SNN variant using smooth firing where applicable, separate threshold/reset and dual-trace gradient paths, prescribed spike surrogate gradients, and RNN input/weight gradient comparisons. Finite-gradient smoke checks are supplementary, not proofs of gradient correctness.
- `test_integration.py`: CNN/layout composition, compilation equivalence, BPTT/online optimizer steps with changing batch sizes, and real weight/state/optimizer checkpoint resume.
- `test_utils.py`: dimension movement, metric conversion, and initialization transforms.

For focused runs:

```sh
python -m pytest tests/core
python -m pytest tests/test_autograd.py
python -m pytest tests/test_snn.py -k threshold
```

Randomness is seeded and restored for each test. Numerical oracle and derivative checks use float64; workflow tests exercise CPU and CUDA explicitly when available. CUDA cases are skipped when unavailable. Tests need no datasets or network access and make no benchmark accuracy or speed claims.

The scalar SNN oracle uses Python arithmetic and does not call production neuron or spike helpers. RNN/LSTM reference mappings account for traceTorch's hidden-first concatenation and LSTM gate ordering. PyTorch's GRUCell is not a direct oracle: traceTorch resets hidden values before the candidate projection and uses its own update convention.

Hard spikes intentionally use a surrogate backward function, so their forward and backward are checked separately. Numerical differentiation of a hard step would not validate that prescribed backward behavior.

All metric-conversion pairs, including half-life, are checked for scalar inputs and tensor inputs on CPU and CUDA when available. Tensor checks cover dtype/device preservation, with analytical gradient checks for half-life conversion in both directions and decay-to-time-constant conversion.
