2. The Ethos of traceTorch
==========================

traceTorch is one of the many PyTorch based SNN libraries that are out there. What sets it apart from the alternatives
is the deliberate design philosophy that prioritizes ergonomics and architectural clarity over needless flexibility. It
abides by the KISS principle: Keep It Stupid Simple (not to be confused with Keep It Simple, Stupid). traceTorch is made
so that models will just work, with as little configuration as necessary for as much power as possible.

At its core, the library employs a two tier architecture to balance computational efficiency with maximum flexibility.
By default, traceTorch presents 32 default layers (``LI``, ``LIB``, ``LIT``, ``LITS``, ``DLI``, ``DLIB``, ``DLIT``, ``DLITS``, ``SLI``, ``SLIB``, ``SLIT``, ``SLITS``, ``RLI``, ``RLIB``,
``RLIT``, ``RLITS``, ``DSLI``, ``DLIB``, ``DSLIT``, ``DSLITS``, ``DRLI``, ``DRLIB``, ``DRLIT``, ``DRLITS``, ``SRLI``, ``SRLIB``, ``SRLIT``,
``SRLITS``, ``DSRLI``, ``DSRLIB``, ``DSRLIT``, ``DSRLITS``), each one written as a standalone class: optimized for its own
specific task. But traceTorch also presents a single, highly configurable ``LeakyIntegrator`` superclass. Each of the default
layers also exists in the ``LeakyIntegrator`` form as a light wrapper, and tests assure that the two versions work identically. The goal of this
split it so have dedicated, optimized layers for the vast majority of applications, while also having the option of dipping
into unique, custom configurations. This single source of truth enables:

- **Extreme Composability**: new dynamics are often obtained by changing the initialization flags.
- **Consistent Mental Model**: one superclass with toggleable flags makes for a simpler mental model than a bunch of
  distinct classes.
- **Easier Extension and Maintenance**: It is easy to modify and maintain the ``LeakyIntegrator`` code to
  create new parameters and features that comply with the rest of the traceTorch ethos.

Another key distinction is the reduction of boilerplate and error-prone state management. Many libraries require manual
initialization, pass, update, detach and reset mechanisms for the model to work. It's understandable why, but that
doesn't make it any comfortable to use. Instead, traceTorch handles hidden states internally:

- **Lazy Initialization**: All states start and reset to ``None`` and are lazily allocated to the correct shape, eliminating
  tensor size mismatch errors.
- **Dimension Agnostic**: Layers focus on a target dimension of the received tensor (defaults to-1, the last dim), so
  that the layers work regardless of your tensor shape.
- **Recursive State Management**: Models initialized with the ``TTModel`` parent class gain access to the ``.detach_states()``
  and ``.zero_states()`` methods which recursively find and apply the respective method no matter how deeply nested the
  traceTorch module is, meaning that you never have to worry about state management.

Further design choices reinforce cleanliness and gradient-friendliness:

- **Rank Based Parameter Initialization**: One single ``*_rank`` argument determines if the parameter is a scalar or vector,
  working as a per-layer or per-neuron parameter.
- **Smooth Constraints for Parameters**: Not all parameters should be unbound. Thresholds and decays are smoothly bound
  by Softplus and Sigmoid respectively, and the actual parameters are stored in logit form of the respective function.
- **Custom Tensors for Parameters**: You don't need to initialize parameters based on a single value, and can instead pass
  in a custom tensor to be the parameter. It's passed through the respective inverse function if necessary, and is automatically
  assigned and managed like any other parameter.
- **Sensible Defaults**: You likely want your parameters to be learnable, per-neuron, set to a sensible value. You
  don't want to write boilerplate arguments every single time you make a model. Thus the defaults arguments are set accordingly:
  layers default to the most powerful configuration, made to look just like any other native PyTorch module.

To manage all the per-layer parameters and hidden states, traceTorch uses the ``TTLayer`` mixin class, which works in tandem
with ``TTModel`` to make everything work effortlessly. ``TTModel`` manages the model-level stuff: recursively calling
``.detach_states()`` and ``.zero_states()`` on the model; while ``TTLayer`` manages the layer-level logic: initializing
hidden states so that they're batch cleared / detached / created, initializing parameters and checking for rank / value / learnability / inverse function,
automating property creation so that ``self.parameter`` returns the parameter passes through ``self.raw_parameter`` through the respective activation
function (sigmoid for decays, softplus for thresholds), helper methods to move a tensor's values around so that it's shape agnostic.

In short, traceTorch exists to make writing, reading, debugging, and most importantly: experimenting, with SNNs in PyTorch
to feel significantly more natural and less frustrating than in existing alternatives, while preserving (and in many cases enhancing)
the expressive power needed for real models and research. It presents both the immediate out-of-the-box variety of a large layer
catalog as well as a unified architecture with more coherent foundation that rewards users who value composition, minimalism, and long-term extensibility.
