The advantages of traceTorch
============================

Like mentioned above, the goal of traceTorch is to make a computationally practical approach to realtime learning based
on sparse rewards. Hence, networks constructed with traceTorch have the following advantages over classic PyTorch ANNs:

- **Recurrent in nature**: Based on SNN principles, traceTorch is recurrent in nature, nonlinearity appearing not
  because of nonlinear functions, but because of the nonlinear nature of the fundamental backbone of SNNs: the Leaky
  Integrate and Fire (LIF) neurons.
- **Constant memory consumption**: As mentioned above, it makes little sense biologically that neurons are somehow able
  to store, traverse and manipulate complex graphs of arbitrary size. Instead, it intuitively makes sense that they
  consume constant memory. traceTorch is built on this principle as well. Despite being recurrent, traceTorch doesn't
  grow its memory consumption regardless of how many forward passes are done.
- **Trace-driven gradients**: traceTorch relies on maintaining an input trace: a running average of the inputs, so that
  it can be possible to reconstruct the "true" average input and hence calculate the approximate output. Thus,
  traceTorch effectively recreates an approximation of the real BPTT autograd graph, without ever needing to actually
  make it or store the history. Through cheap, local updates, traceTorch effectively mimics BPTT by compressing the
  entire history into one singular forward pass, with more value attributed to temporally recent events and inputs.
- **Intermittent, online learning**: With traceTorch being reliant entirely on local calculations, and never needing to
  build a full graph, the backward pass can be invoked at any point in time. Backward passes, reliant on small, local
  per-layer graphs, are also very memory-friendly, consuming a fixed amount of time no matter how many forward passes
  were committed before.