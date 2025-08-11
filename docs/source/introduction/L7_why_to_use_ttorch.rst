What is traceTorch useful for?
==============================

Admittedly, traceTorch fills quite a specific niche. Online, biologically plausible learning, bordering on the concept
of Single Lifetime Learning (SLL), is hardly something that serves a practical purpose, and is instead designed with the
intent of experiment and research. It is designed for playing around, experimenting. Think of it as a way to make
artificial creatures, and then training them in a similar fashion to how real creatures are trained. To facilitate this,
traceTorch is primarily focused around:

1. A suite of nn modules, akin to the PyTorch counterpart, designed to create the networks.
2. A suite of loss modules, designed to generate the initial learning signal, the derivative of the loss w.r.t. the
   model's outputs.
3. A suite of plot modules, designed to visualize the hidden states and anything else of frequent use or interest.