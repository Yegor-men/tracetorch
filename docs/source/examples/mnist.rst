MNIST Examples
==============

The MNIST examples live in ``examples/mnist``. They all train three comparable models: an SNN, a GRU-based RNN, and an
S6-based SSM. This makes the examples useful for seeing how traceTorch layers can be swapped while the training loop
stays nearly identical.

Setup:

.. code-block:: bash

    cd examples/mnist
    pip install -r requirements.txt

Rate-coded MNIST
----------------

Run:

.. code-block:: bash

    python rate_coded.py

The rate-coded script presents the same MNIST image for several timesteps. At each timestep, the input image is converted
to a binary sample:

.. code-block:: python

    for t in range(num_timesteps):
        spk_image = torch.bernoulli(image)
        output = model(spk_image)
        loss = loss_fn(output, label)
        running_loss += loss

The SNN model is:

.. code-block:: python

    class RateSNN(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                tt.snn.LIB(128, beta=torch.rand(128), threshold=torch.rand(128)),
                nn.Linear(128, 128),
                tt.snn.LIB(128, beta=torch.rand(128), threshold=torch.rand(128)),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.net(x)

The important traceTorch mechanics are ``model.zero_states()`` before each image batch and repeated forward calls over
the timestep loop.

Sequential MNIST
----------------

Run:

.. code-block:: bash

    python sequential.py

The sequential script turns each image into a sequence of patches. A ``4x4`` patch becomes one timestep with 16 input
features. The model sees a scrambled sequence of local image observations and must produce a final classification.

The SNN uses recurrent binary layers:

.. code-block:: python

    self.net = nn.Sequential(
        nn.Linear(kernel_size ** 2, 128),
        tt.snn.RLIB(128, beta=torch.rand(128), gamma=torch.rand(128), threshold=torch.rand(128)),
        nn.Linear(128, 128),
        tt.snn.RLIB(128, beta=torch.rand(128), gamma=torch.rand(128), threshold=torch.rand(128)),
        nn.Linear(128, 10),
    )

This is a good example of when ``R`` layers make sense: the output at one patch can influence the membrane update at
the next patch.

Noisy MNIST
-----------

Run:

.. code-block:: bash

    python noisy.py

The noisy script repeatedly corrupts the same input image and trains the model on several noisy observations:

.. code-block:: python

    noise_level = torch.rand_like(image) ** 0.5
    noisy_image = image * noise_level + (torch.randn_like(image) + 0.5) * (1 - noise_level)
    output = model(noisy_image)

This example is useful for understanding traceTorch as a temporal evidence accumulator. The model receives multiple
imperfect observations and can use its hidden states to build a more stable representation over time.

Reading the plots
-----------------

Each script plots training and evaluation loss/accuracy for the SNN, RNN, and SSM variants. The examples are intended
for comparison and code clarity, not for state-of-the-art MNIST results.
