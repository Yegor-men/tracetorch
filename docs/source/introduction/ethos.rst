The Ethos of traceTorch
=======================

traceTorch initially begun as a personal project, and subsequently, the user-end was always at the forefront. Thus, it
follows an opinionated and strict set of rules: the deliberate design philosophy that prioritizes ergonomics over needless
flexibility, while being mathematically and conceptually clean in the backend. traceTorch abides by the KISS principle:
Keep It Stupid Simple (not to be confused with Keep It Simple, Stupid). It is made so that models will just work, with
as little configuration as necessary for as much power as possible.

Thus, traceTorch presents two core modules: ``tracetorch.Model`` and ``tracetorch.Layer`` which manage the model and layer level boilerplate.
They work in tandem: ``tt.Layer`` handles everything to do within the layer, while ``tt.Model`` recursively searches for layers
inside it. This means that from the user end, the only change you need to do is to inherit from ``tt.Model`` rather than ``nn.Module``,
and the rest is handled automatically, and the architecture looks no different to classic PyTorch.


The Model Module
----------------
Let's make a simple dummy model and see what methods we have available:

.. code-block:: python

    import torch
    import tracetorch as tt

    class Model(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768, 128, bias=False),
                tt.snn.LIB(128),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = Model().to("cuda")

With the dummy model made above, we can now easily manage all the hidden states, no matter how deeply buried they are (in the current example, it's just the LIB module):

- ``model.zero_states()`` to set all the states to ``None``, so that they get dynamically created in the next forward pass as to avoid size mismatch errors
- ``model.detach_states()`` to set each state to a detached version of itself, used for truncated backpropagation through time or for online learning

Both these methods work thanks to ``tt.Model``'s recursive search: it only looks for other traceTorch models inside it,
and traceTorch layers to actually call the method on. This means that traceTorch methods don't conflict with your other code:
you can integrate it anywhere without consequence.

Using this recursive search feature, we're also able to save and load the hidden states with ``model.save_states()`` and
``model.load_states()`` in the exact same way that we'd save the model's parameters (check out :doc:`this tutorial <../tutorials/save_load>` on how to do so).

Many of the traceTorch layers use various functions for numerical stability and gradient cleanliness. These functions are
applied in each forward pass, so it would be nice if we could optimize a trained model and rid ourselves of the needless computations.
Thanks again to the recursive search, we can simply call ``model.TTcompile()`` to compile a model and ``model.TTdecompile()``
to get it back into training mode (check out :doc:`this tutorial <../tutorials/compile_decompile>` on how to do so).

All in all, managing the hidden states of a model is incredibly simple and adds zero boilerplate. These 6 methods alone
are enough for you to begin creating and training your own models.


The Layer Module
----------------
Whereas the model module handles the boilerplate of state management, the layer module handles the boilerplate of parameter
initialization and local state management. Knowing how to use ``tt.Layer`` is not necessary unless if you want to make your
own traceTorch compliant layers.

It's simplest to understand ``tt.Layer`` if we directly look at the source code of some layer. For example, the aforementioned LIB:

.. code-block:: python

    class LIB(tt.Layer):
        def __init__(
                self,
                num_neurons: int,
                beta: Union[float, torch.Tensor] = 0.9,
                threshold: Union[float, torch.Tensor] = 1.0,
                bias: Union[float, torch.Tensor] = 0.0,
                dim: int = -1,
                beta_rank: Literal[0, 1] = 1,
                threshold_rank: Literal[0, 1] = 1,
                bias_rank: Literal[0, 1] = 1,
                learn_beta: bool = True,
                learn_threshold: bool = True,
                learn_bias: bool = True,
                spike_fn=functional.sigmoid4x,
                quant_fn=nn.Identity(),
        ):
            super().__init__(num_neurons, dim)

            self._initialize_state("mem")
            self._register_decay("beta", beta, beta_rank, learn_beta)

            self.spike_fn = spike_fn
            self.quant_fn = quant_fn
            self._register_threshold("threshold", threshold, threshold_rank, learn_threshold)
            self._register_bias("bias", bias, bias_rank, learn_bias)

        def forward(self, x):
            self._ensure_states(x)
            x = self._to_working_dim(x)

            mem = self._to_working_dim(self.mem)
            mem = mem * self.beta + x

            spike_prob = self.spike_fn(mem - self.threshold + self.bias)
            spikes = self.quant_fn(spike_prob)

            mem = mem - spikes * self.threshold

            spikes = self._from_working_dim(spikes)
            self.mem = self._from_working_dim(mem)

            return spikes

We see the full range of available methods used here:

- ``_initialize_state`` correctly initializes a hidden state so that traceTorch can manage it
- ``_register_decay`` / ``_register_threshold`` / ``_register_bias`` are wrappers around a ``_register_parameter`` method, which automatically handles the rank, learnability and value (or a custom tensor) for initialization of a parameter that can be optimized with the ``.TTcompile()`` method, as some of these parameters need an activation function in order to be actually usable
- ``_ensure_states`` makes all hidden states attain the shape of some target tensor, with the exception of the target ``dim`` dimension which is set to ``num_neurons`` instead (the two arguments used for the superclass initialization)
- ``_to_working_dim`` / ``_from_working_dim`` move a tensor's dimensions around so that the layer works on the target ``dim``, rather than just the final dimension

These methods thus allow traceTorch layers to work in any situation and architecture, significantly cutting down on the
boilerplate necessary to get a working model.


Summary
-------

In short, traceTorch exists to make writing, reading, debugging, and most importantly: experimenting, with recurrent
networks in PyTorch to feel significantly more natural and less frustrating than in existing alternatives, while preserving
(and in many cases enhancing) the expressive power needed for real models and research. It ultimately rewards users who
value composition, minimalism, and long-term extensibility.