L1. The TTModule
================

In ``traceTorch``, the layers are made to maximally integrate with ``PyTorch``. But unlike typical ``PyTorch`` layers
which are stateless, SNN layers are inherently stateful, which means that we also need to add management of the hidden
states. Namely, we want a quick and easy way to clear them and ready for the next batch; and to detach them for
truncated BPTT or online learning.

``traceTorch`` presents `tracetorch.snn.TTModule`, a replacement to ``pytorch.nn.Module`` which provides you with two
methods: ``.zero_states()`` which sets the states to ``None`` and readies them for the next batch, and ``.detach_states()``
which detaches all the hidden states. Both these methods recursively search not just the attributes in the class instances,
but _all_ the attributes' attributes, and so on. It goes through the entire tree of the class instance, meaning that it
will find _all_ the hidden states no matter how far they've been hidden, no matter the architecture. Just initialize
your model to use ``snn.TTModule`` instead of ``nn.Module`` and you're good to go.
