T2. Nonmonotonic MNIST
======================

The previous task was simple, in that it was effectively a rehash of classification. The only challenge that was added
was that the input was noisy, and the model needed to accumulate enough of it to create an accurate representation of
the true image.

However, we want to make the task harder, nonmonotonic. The model has to learn temporal dynamics and can no longer
blindly accumulate charge and automatically reduce the task, it has to fundamentally learn the concept of time and
stateful dynamics. We shall do this by changing the MNIST image to a time-series data, a kernel sliding across the image,
and it's what the kernel sees that the model will see. The model must hence learn how the bit sequences map to the numbers,
and must learn hence _learn_ to turn the nonmonotonic task into a monotonic one via the temporal and stateful dynamics.