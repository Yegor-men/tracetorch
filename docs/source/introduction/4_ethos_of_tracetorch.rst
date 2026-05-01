4. The Ethos of traceTorch
==========================

traceTorch is a unified recurrent neural network library for PyTorch that rethinks how recurrent networks are built from
the ground up. What sets it apart from alternatives is the deliberate design philosophy that prioritizes ergonomics and
architectural clarity over needless flexibility. It abides by the KISS principle: Keep It Stupid Simple (not to be confused with Keep It Simple, Stupid).
traceTorch is made so that models will just work, with as little configuration as necessary for as much power as possible.

At its core, the library presents a unified approach to recurrent networks, encompassing both traditional RNN architectures
and Spiking Neural Networks (SNNs). The main focal points can be found in ``tt.rnn`` and ``tt.snn``:

- **tt.rnn**: Traditional recurrent architectures including ``SimpleRNN``, ``LSTM``, ``GRU``, and modern State Space Models like ``Mamba``
- **tt.snn**: A comprehensive collection of 32 specialized SNN layer types built around the Leaky Integrator, offering diverse dynamics through modular naming

The fundamental innovation is the enforcement of one simple rule that should have been the default all along: **hidden states stay hidden**.
But that's not to say they're inaccessible. On the contrary, traceTorch is designed with ergonomics at the forefront,
making state management easier than ever. Hidden states are lazily created in the forward pass, work with any target dimension,
and are easy to clear, detach, and even save and load.

Another key distinction is the reduction of boilerplate and error-prone state management. Many libraries require manual
initialization, pass, update, detach and reset mechanisms for the model to work. Instead, traceTorch handles hidden states internally:

- **Lazy Initialization**: All states start and reset to ``None`` and are lazily allocated to the correct shape, eliminatingtensor size mismatch errors.
- **Dimension Agnostic**: Layers focus on a target dimension of the received tensor (defaults to -1, the last dim), so that the layers work regardless of your tensor shape.
- **Recursive State Management**: Models inheriting from ``tt.Model`` gain access to ``.detach_states()`` and ``.zero_states()`` methods which recursively find and apply the respective method no matter how deeply nested the traceTorch module is, meaning that you never have to worry about state management.

For SNN layers specifically, traceTorch employs a two-tier architecture to balance computational efficiency with maximum flexibility. By default, it presents 32 specialized layers (``LI``, ``DLI``, ``SLI``, ``DSLI``, ``LIB``, ``DLIB``, ``SLIB``, ``RLIB``, ``DSLIB``, ``DRLIB``, ``SRLIB``, ``DSRLIB``, ``LIT``, ``DLIT``, ``SLIT``, ``RLIT``, ``DSLIT``, ``DRLIT``, ``SRLIT``, ``DSRLIT``, ``LITS``, ``DLITS``, ``SLITS``, ``RLITS``, ``DSLITS``, ``DRLITS``, ``SRLITS``, ``DSRLITS``), each written as a standalone class optimized for its specific task. But traceTorch also presents a single, highly configurable ``LeakyIntegrator`` superclass. This dual approach enables:

- **Extreme Composability**: new dynamics are often obtained by changing the initialization flags.
- **Consistent Mental Model**: one superclass with toggleable flags makes for a simpler mental model than a bunch of distinct classes.
- **Easier Extension and Maintenance**: It is easy to modify and maintain the ``LeakyIntegrator`` code to create new parameters and features.

Further design choices reinforce cleanliness and gradient-friendliness across all layer types:

- **Rank Based Parameter Initialization**: One single ``*_rank`` argument determines if the parameter is a scalar or vector, working as a per-layer or per-neuron parameter.
- **Smooth Constraints for Parameters**: Not all parameters should be unbound. Thresholds and decays are smoothly bound by Softplus and Sigmoid respectively, and the actual parameters are stored in logit form of the respective function.
- **Custom Tensors for Parameters**: You don't need to initialize parameters based on a single value, and can instead pass in a custom tensor to be the parameter. It's passed through the respective inverse function if necessary, and is automatically assigned and managed like any other parameter.
- **Sensible Defaults**: You likely want your parameters to be learnable, per-neuron, set to a sensible value. You don't want to write boilerplate arguments every single time you make a model. Thus the defaults arguments are set accordingly: layers default to the most powerful configuration, made to look just like any other native PyTorch module.

To manage all the per-layer parameters and hidden states, traceTorch uses the ``tt.Layer`` class, which works in tandem with ``tt.Model`` to make everything work effortlessly. ``tt.Model`` manages the model-level stuff: recursively calling ``.detach_states()`` and ``.zero_states()`` on the model; while ``tt.Layer`` manages the layer-level logic: initializing hidden states so that they're batch cleared / detached / created, initializing parameters and checking for rank / value / learnability / inverse function, automating property creation so that ``self.parameter`` returns the parameter passes through ``self.raw_parameter`` through the respective activation function (sigmoid for decays, softplus for thresholds), helper methods to move a tensor's values around so that it's shape agnostic.

In short, traceTorch exists to make writing, reading, debugging, and most importantly: experimenting, with recurrent networks in PyTorch to feel significantly more natural and less frustrating than in existing alternatives, while preserving (and in many cases enhancing) the expressive power needed for real models and research. It presents both the immediate out-of-the-box variety of a large layer catalog as well as a unified architecture with more coherent foundation that rewards users who value composition, minimalism, and long-term extensibility.