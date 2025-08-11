Disadvantages of traceTorch
===========================

traceTorch is not an upgrade or extension to base PyTorch, it is fundamentally designed as a replacement for PyTorch
given very specific, admittedly unusual requirements centred around biological plausibility. Subsequently, you may face
the following challenges:

- **Not a framework to make SNNs**: While traceTorch _could_ be used as a method of making SNNs, if your goal is to
  simply use SNN layers instead of the usual PyTorch layers, traceTorch is not recommended. Libraries such as
  [snntorch](https://github.com/jeshraghian/snntorch) are built specifically around this, they directly incorporate
  SNN layers with the PyTorch autograd in a differentiable manner.
- **Gradient approximations**: The fundamental principle of the backward pass in traceTorch is the reconstruction of
  average outputs based on the learnable parameters and the average input. It is an approximation, not an actual 1:1
  reconstruction or mathematically equivalent substitute. Subsequently, performance will be worse than if using true
  backpropagation.
- **Can't be incorporated directly into an existing classic architecture**: While traceTorch can be used alongside
  classic PyTorch models (such as an autoencoder which can pass it's encoded latent to the traceTorch model), traceTorch
  layers can't be directly interweaved into classic modules like torch.nn.Sequential, the very nature of traceTorch
  making graph free forward passes while being recurrent doesn't consolidate with the classic implementation and
  function of recurrent models which need to store the history.
- **No practical benefit compared to alternatives**: If your goal is to construct an ANN and deploy it, traceTorch is of
  no help. Fundamentally, a simple way to think of the principle of traceTorch is that it compresses a theoretically
  massive autograd graph of multiple recurrent forward passes into just one, all the while doing it on the fly. But
  compression leads to loss, it's simply not possible to gain quality or accuracy by compressing and losing information.
