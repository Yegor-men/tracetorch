T0. Memorizing Patterns
=======================

As the first task and challenge, we want to see if a traceTorch model is able to memorize a specific sequential pattern. This verifies that our state mechanism, compilation mechanisms, and backpropagation logic are functioning correctly over a basic temporal problem.

The Setup
---------

In this tutorial (modeled heavily on ``tests/test_core.py``), we generate a random 2D image and assign it a fixed random label. We then force the model to continuously observe this same exact image over many timesteps and see if it can memorize the output. 

.. code-block:: python

    import torch
    from torch import nn
    import tracetorch as tt
    from tracetorch import snn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # We will simulate a batch size of 2, 3 color channels, 10x10 resolution
    b, c, h, w = 2, 3, 10, 10
    
    random_feature = torch.rand(b, c, h, w).to(device)
    random_label = torch.eye(b).to(device) # Just generating a random target

Model Architecture
------------------

We define a simple Multi-Layer Perceptron (MLP) mixed with some Spiking layers. 

.. code-block:: python

    class SimpleSNN(tt.Model):
        def __init__(self, c, n_labels):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv2d(c, 16, 3),
                snn.LIB(16, 0.9, 1.0, dim=-3),
                nn.Flatten(),
                nn.Linear(16 * 8 * 8, 16), # 10x10 -> 8x8 due to 3x3 conv
                snn.RLIB(16, 0.9, 0.9, 1.0),
                nn.Linear(16, n_labels),
                snn.LI(n_labels, 0.9),
            )

        def forward(self, x):
            return self.mlp(x)

    model = SimpleSNN(c, b).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.functional.mse_loss

Training Over Time
------------------

To train the network, we must repeatedly pass our static feature into the network over time. The recurrent state dynamically builds over the sequence loop.

.. code-block:: python

    num_epochs = 100
    num_timesteps = 10

    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        
        # Reset the hidden membrane states
        model.zero_states()
        
        for step in range(num_timesteps):
            # The model processes the same static feature at every timestep
            model_output = model(random_feature)
            
        # We compute the loss on the final output
        loss = loss_fn(model_output, random_label)
        loss.backward()
        optimizer.step()

If the loss smoothly converges to 0, it indicates that our architecture and the ``tracetorch`` recurrent states successfully propagate gradients through time and can accurately map the input to the required output.