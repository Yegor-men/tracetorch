1. What is an RNN?
==================

It is a well known fact that neural networks are conceptually inspired by the structure and function of biological brains.
After all, that's where the very name comes from. Conceptually, the idea is that each neuron functions as a small computation
unit, and together, chaining and parallelizing many of these computational units results in a meaningful network that can
generalize and perform some task, just like our brains. Thus arises the name: Artificial Neural Network (ANN).

But how accurate is this with respect to real biology and neuroscience? Well, not really. Most notably would be the fact
that ANNs are typically stateless. That is to say that they are static functions, they don't create or manipulate an explicit
"hidden state" that somehow meaningfully represents the current state of the data. Some ANNs do emulate the concept of a state,
transformer based LLMs for example: looking at the fully attended token but before the prediction of the next one, an
argument could be raised that it is effectively a state of sorts evolving throughout the sequence. But this isn't really the same thing.

Thus comes the rise of Recurrent Neural Networks (RNNs): models that do in have one or more hidden states. They learn to
create, manipulate and use one or more of these hidden states, the core goal being that this state somehow sufficiently describes
everything the model needs. Going back to the transformer LLM example, the fully attended token can be considered a state, as
it sufficiently encodes all the necessary information to predict the next token.

The earliest RNNs by modern standards would be the Elman and Jordan networks: networks that explicitly have an internal
memory and process data one timestep at a time, updating that internal memory. But they struggle a lot with learning long sequences.
Later came the breakthrough with the Long Short-Term Memory (LSTM) networks which could suddenly learn for much longer sequences.
Even more recently came the invention of the Gated Recurrent Unit (GRU), a simplification to LSTMs while maintaining performance.
All the  more recently came interesting developments in State Space Models (SMMs), such as the advent of Mamba, with an even longer and better memory.

There is a great many various RNN architectures, but the key concept boils down to the idea of having a kind of hidden state:
recording data into the hidden state as to access it later and being able to extract the necessary data from it.