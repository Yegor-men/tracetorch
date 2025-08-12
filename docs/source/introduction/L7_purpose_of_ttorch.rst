7. What is the purpose of traceTorch?
=====================================

So, with the positives and negatives in consideration, what even is the purpose of ``traceTorch``? Admittedly, it fills
quite the specific niche. Online, biologically plausible learning, bordering on the concept of Single Lifetime Learning
(SLL), is hardly something that serves a practical purpose, and is instead designed with the intent of playing around,
experimenting, researching. Think of it as a way to make artificial creatures, and then training them in a similar
fashion to how real creatures are trained. ``traceTorch`` in it's entirety is designed around this workflow:
creating the model, training it, inspecting and assessing it.

It's not a 1:1 recreation of biological neurons, but that was never the intent. It's not an in-place replacement of
conventional PyTorch models, because it has the biological restrictions applied. It's a kind of amalgamation of the two,
aiming to create biological-esque networks in a computationally feasible manner.

With the introduction out of the way, it's recommended that you read through some of the tutorials, at the very least
:doc:`Tutorial 0 <../tutorials/T0_terminology>` to become familiar with the terminology used in ``traceTorch``.
