Quickstart
==========

This page builds the smallest useful traceTorch training loop. The goal is not accuracy; it is to show the shape of a
traceTorch model.

The model
---------

traceTorch models are ordinary PyTorch modules with one important change: inherit from ``tt.Model`` instead of
``nn.Module``. That gives the model recursive state-management methods.

.. code-block:: python

    import torch
    from torch import nn
    import tracetorch as tt


    class Net(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                tt.snn.LIB(num_neurons=128),
                nn.Linear(128, 10),
                tt.snn.LI(num_neurons=10),
            )

        def forward(self, x):
            return self.net(x)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)

``tt.snn.LIB`` is a leaky integrate-and-binary-fire layer. With the default ``quant_fn=nn.Identity()``, it returns a
smooth firing value rather than a hard discrete spike. ``tt.snn.LI`` is a continuous leaky integrator, useful as a simple
readout trace.

The timestep loop
-----------------

traceTorch layers process one timestep per forward call. If an input has 20 timesteps, the outer training loop calls the
model 20 times.

.. code-block:: python

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.functional.cross_entropy

    for image, label in train_dataloader:
        image = image.to(device)
        label = label.to(device)

        model.train()
        model.zero_grad()
        model.zero_states()

        running_output = 0
        for _ in range(20):
            x_t = torch.bernoulli(image)
            running_output = running_output + model(x_t)

        output = running_output / 20
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

The important line is ``model.zero_states()``. It resets all traceTorch hidden states to ``None`` so they will be lazily
created with the correct batch shape on the first timestep.

Online learning
---------------

If you want gradients to stop at each timestep, call ``detach_states()`` inside the timestep loop.

.. code-block:: python

    for t in range(num_timesteps):
        output = model(x[t])
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.detach_states()

This is useful for online learning and truncated backpropagation through time. If you want full backpropagation through
the whole sequence, do not detach until after the sequence.

Working with images
-------------------

Layers operate on the dimension given by ``dim``. The default is ``-1``, which is natural for MLPs. For image channels,
use ``dim=-3``.

.. code-block:: python

    layer = tt.snn.LIB(num_neurons=32, dim=-3)
    x = torch.rand(16, 32, 28, 28)
    y = layer(x)
    print(y.shape)
    # torch.Size([16, 32, 28, 28])

Next steps
----------

Read :doc:`introduction/index` to understand the design choices, then follow :doc:`examples/mnist` for complete runnable
MNIST scripts.
