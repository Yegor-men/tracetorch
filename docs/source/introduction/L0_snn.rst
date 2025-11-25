L0. What is an SNN?
===================

SNN, shorthand for Spiking Neural Network, is a subset of neural nets that aim to recreate, or at least more closely
follow, the neuron dynamics we observe in biology, where neurons typically function on discrete signals: all or nothing
"spike" events. Computationally, this typically means representing layer outputs as exclusively 0s or 1s.

``traceTorch`` (typically) follows these principles, in a way that easily integrates with your existing ``PyTorch`` code.
