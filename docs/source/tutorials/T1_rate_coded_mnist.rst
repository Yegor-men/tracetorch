T1. Rate-Coded MNIST
====================

The rate-coded MNIST example is the canonical "Hello World" of Spiking Neural Networks (SNNs). Unlike standard Artificial Neural Networks (ANNs) which take a fixed vector of intensity values representing a pixel, rate-coded models convert pixel intensity into a probability of a spike occurring at any given timestep.

While this is traditionally an SNN task, traceTorch allows us to test **SNNs**, **RNNs**, and **SSMs** on the exact same task with identical architectures.

The Complete Code
-----------------

The code for this tutorial is located in ``examples/mnist/rate_coded.py``. Below we will break down the essential components.

Model Definition
----------------

We can define three distinct models utilizing traceTorch's unified layer ecosystem:

.. code-block:: python

    import tracetorch as tt
    from torch import nn

    class RateSNN(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                tt.snn.LIB(128, dim=-1),
                nn.Linear(128, 10),
                tt.snn.LI(10)
            )
        def forward(self, x): return self.net(x)

    class RateRNN(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                tt.rnn.GRU(128, 128, dim=-1),
                nn.Linear(128, 10)
            )
        def forward(self, x): return self.net(x)

    class RateSSM(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                tt.ssm.S6(128, 16),
                nn.Linear(128, 10)
            )
        def forward(self, x): return self.net(x)

Training Loop 
-------------

Since traceTorch enforces a 1-timestep-at-a-time methodology for *all* its recurrent layers, the training loop remains absolutely identical regardless of whether you instantiate the SNN, RNN, or SSM.

.. code-block:: python

    for image, label in train_dataloader:
        model.zero_grad()
        
        # Reset the hidden states for the new batch
        model.zero_states()  

        running_loss = 0.0
        
        # Iterate over time
        for t in range(num_timesteps):
            # Convert intensity into spike probabilities
            spk_image = torch.bernoulli(image)
            
            output = model(spk_image)
            loss = loss_fn(output, label)
            running_loss += loss

        running_loss = running_loss / num_timesteps
        running_loss.backward()
        optimizer.step()

We can then plot the accuracies to see how the SNN compares against its RNN and SSM counterparts in a rate-coded paradigm.