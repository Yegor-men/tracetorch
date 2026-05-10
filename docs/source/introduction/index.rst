Introduction
============

The introduction is the conceptual part of the documentation. Read it before the examples if traceTorch feels unusual.

traceTorch is intentionally small and opinionated: layers own their hidden states, models manage those states
recursively, and sequences are processed one timestep at a time. Once that model clicks, the rest of the library feels
like ordinary PyTorch.

Recommended order:

1. :doc:`ethos` explains why traceTorch exists and what problems it chooses to solve.
2. :doc:`stateful_models` explains hidden states, timestep loops, reset/detach, and state persistence.
3. :doc:`snns` explains SNN dynamics, surrogate outputs, quantization, and the 32-layer naming scheme.
4. :doc:`layers` gives a compact map of the SNN, RNN, SSM, core, functional, and plotting modules.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    ethos
    stateful_models
    snns
    layers
