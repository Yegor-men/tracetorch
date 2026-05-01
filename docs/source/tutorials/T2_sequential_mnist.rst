T2. Sequential MNIST
====================

While rate-coded MNIST tests the ability to accumulate information over time, **Sequential MNIST** tests the ability of a recurrent model to remember past context.

In Sequential MNIST, the 2D image is unrolled into a 1D sequence of pixels. To make the task significantly harder, we randomly shuffle the sequence permanently using a fixed permutation. This forces the model to memorize distant pixel relationships instead of relying on spatial locality.

The Complete Code
-----------------

The complete, runnable code is available at ``examples/mnist/sequential.py``.

Model Definition
----------------

We will benchmark three distinct traceTorch models (an SNN, an RNN, and an SSM) to see how they handle long sequences. 

*Note: While native SSMs like Mamba are often prized for their parallelization over sequences, traceTorch currently forces SSMs to operate one timestep at a time (just like its SNNs and RNNs) to maintain architectural uniformity across the library. This allows mixing and matching layers effortlessly.*

.. code-block:: python

    import tracetorch as tt
    from torch import nn

    class SeqSNN(tt.Model):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(kernel_size ** 2, 128)
            self.layer = tt.snn.RLIB(128, dim=-1)
            self.dec = nn.Linear(128, 10)
        def forward(self, x): return self.dec(self.layer(self.enc(x)))

    class SeqSSM(tt.Model):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(kernel_size ** 2, 128)
            self.layer = tt.ssm.S6(128, 16)
            self.dec = nn.Linear(128, 10)
        def forward(self, x): return self.dec(self.layer(self.enc(x)))

Training Loop
-------------

Unlike rate-coded tasks which accumulate loss at every timestep, models on sequential tasks often only evaluate the final output to determine the classification.

.. code-block:: python
    
    for seq, label in train_dataloader:
        model.zero_grad()
        
        # Reset memory state
        model.zero_states()

        final_output = None
        
        # Feed the sequence patch by patch
        for t in range(seq.size(0)):
            final_output = model(seq[t])

        # Evaluate the loss ONLY on the final output prediction
        loss = loss_fn(final_output, label)
        loss.backward()
        optimizer.step()

This demonstrates traceTorch's unified capability: whether you are using spiking neurons (SNNs), classic recurrent layers (RNNs), or modern State Space Models (SSMs), the underlying state management (via ``model.zero_states()``) remains identically elegant.