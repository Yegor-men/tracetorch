0. Terminology used throughout traceTorch
=========================================

Here is contained all the terminology you may come across while using ``traceTorch``, listed by the importance of
needing to know them. This isn't necessary but is helpful. Feel free to check back here if there is something you find
that you don't understand.

**SNN mechanics**:

- ``spike``: when a neuron fires, outputting a 1.
- ``spike train``: spikes and their absence over time, recorded for some chosen neurons.

**Learning / backward pass**:

- ``downstream``: any layer that is calculated later in the forward pass, typically referring to the ``n+1`` th layer in
  the forward pass when looking at the ``n`` th layer.
- ``upstream``: any layer that is calculated earlier in the forward pass, typically referring to the ``n-1`` th layer in
  the forward pass when looking at the ``n`` th layer.
- ``learning signal``: the derivative of the loss w.r.t. some neuron or layer's output.

**Trace mechanics**:

- ``trace``: a running average of sorts. At each timestep, it's multiplied by some decay factor ``d``, and to it is
  added the input ``i``.
- ``average input``: the calculated average input ``i`` given some known decay factor ``d`` and trace ``t``, using the
  relation ``i=t(1-d)``.
- ``stabilized trace``: the calculated trace ``t`` given some known factor ``d`` and average input ``i``, using the
  relation ``t=i/(1-d)`` by rearranging ``i=t(1-d)``
