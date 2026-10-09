Layer Map
=========

This page is a compact map of the library. Use it when you know roughly what you want but not which module or layer name
to reach for.

SNN layers
----------

``tt.snn`` is the main part of traceTorch. It contains 32 leaky-integrator-based layers.

The output families are:

``LI``
    Continuous leaky integrator output. No threshold, no firing, no reset. Variants: ``LI``, ``DLI``, ``SLI``, ``DSLI``,
    ``LIEMA``, ``DLIEMA``, ``SLIEMA``, ``DSLIEMA``.

``LIB``
    One-sided binary-style firing. Uses one positive threshold. Variants: ``LIB``, ``DLIB``, ``SLIB``, ``RLIB``,
    ``DSLIB``, ``DRLIB``, ``SRLIB``, ``DSRLIB``.

``LIT``
    Ternary-style firing with positive and negative thresholds. Variants: ``LIT``, ``DLIT``, ``SLIT``, ``RLIT``,
    ``DSLIT``, ``DRLIT``, ``SRLIT``, ``DSRLIT``.

``LITS``
    Ternary-style firing with separate positive and negative output scales. Variants: ``LITS``, ``DLITS``, ``SLITS``,
    ``RLITS``, ``DSLITS``, ``DRLITS``, ``SRLITS``, ``DSRLITS``.

The prefixes are mechanical:

``D``
    Dual positive/negative traces.

``S``
    Synaptic input trace before the membrane.

``R``
    Recurrent trace of the previous output.

Start with ``LIB`` for ordinary SNN experiments, ``LI`` for continuous readout, ``LIT`` for signed events, and ``LITS``
when signed events need separate magnitudes.

RNN layers
----------

``tt.rnn`` contains classic recurrent layers with traceTorch state management:

``SimpleRNN``
    A tanh Elman-style recurrent layer.

``LSTM``
    A long short-term memory layer with hidden and cell states.

``GRU``
    A gated recurrent unit with reset and update gates.

These layers process one timestep per forward call. They are useful when you want conventional RNN dynamics but still
want ``tt.Model.reset_states()``, ``detach_states()``, and state saving.

Core layers
-----------

``tt.Layer`` is the base class for traceTorch layers. It provides:

* state registration and lazy state initialization;
* recursive reset/detach behavior through ``tt.Model``;
* constrained parameter registration;
* compile/decompile support.

Most users do not need to subclass ``tt.Layer`` immediately. Read :doc:`../tutorials/custom_layer` when you want to
create a new traceTorch-compatible layer.

Helpers
-------

``tt.utils`` provides decay conversion and the ``MoveDim`` wrapper for explicit layouts.
``tt.inverse_fn`` provides initialization transforms for constrained parameters.
``tt.snn.spike_fn`` provides deterministic, stochastic and smooth firing functions.
Plotting utilities live in ``examples/plotting.py``, not the package.
