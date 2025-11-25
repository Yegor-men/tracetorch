T1. Monotonic MNIST
===================

A slight step up from rote memorization is to see if the model can generalize. However, we will still use a monotonic
task at the core.

Monotonic functions are those where "more input" = "more outupt". Monotonic here meaning that the model simply needs to
accumulate charge, and the more that it blindly accumulates, the more accurate it's result will be. In the case of MNIST,
we will flash a bernoulli spike of the entire number at each timestep. It's not hard to imagine that the more that the
model just accumulates charge, the more accurately it will see the underlying image and hence reduce the task to simple
classification.