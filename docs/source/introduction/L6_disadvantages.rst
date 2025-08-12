6. The disadvantages of traceTorch
==================================

With the advantages out of the way, it's time to discuss the disadvantages, or rather, the things that ``traceTorch``
cannot do or struggles with, the areas where ``traceTorch`` is inferior to conventional models.

#. **Gradient approximations**: The fundamental principle of the backward pass in ``traceTorch`` is the reconstruction of
   average outputs based on the learnable parameters and the average input. It is an approximation, not an actual 1:1
   reconstruction or mathematically equivalent substitute. Subsequently, performance will be worse than if using true
   backpropagation.
#. **Can't be incorporated directly into an existing classic architecture**: While traceTorch can be used alongside
   classic PyTorch models (such as an autoencoder which passes it's encoded latent to the traceTorch model), traceTorch
   layers can't be directly interweaved into classic modules like ``torch.nn.Sequential()``, the very nature of
   ``traceTorch`` making graph free forward passes while being recurrent doesn't consolidate with the classic
   implementation and function of recurrent models which need to store the history.
#. **No practical benefit compared to alternatives**: If your goal is to construct an ANN and deploy it, traceTorch is of
   no help. Fundamentally, a simple way to think of the principle of traceTorch is that it compresses a theoretically
   massive autograd graph of multiple recurrent forward passes into just one, all the while doing it on the fly. But
   compression leads to loss, it's simply not possible to gain quality or accuracy by compressing and losing information.

If you are interested in SNN networks which *do* utilize autograd in the conventional sense, being able to be swapped in
with conventional PyTorch layers and autograd graph constructions, it's suggested that you use other libraries, such as
`snntorch <https://github.com/jeshraghian/snntorch>`__ instead.
