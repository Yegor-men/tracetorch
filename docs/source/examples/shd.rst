Heidelberg Digits
=================

The Heidelberg Digits example lives in ``examples/heidelberg_digits/main.py``. It uses the Spiking Heidelberg Digits
dataset through Tonic and trains SNN, GRU, and S6 models on framed event data.

Setup:

.. code-block:: bash

    cd examples/heidelberg_digits
    pip install -r requirements.txt
    python main.py

Dataset
-------

The script converts event streams into frames:

.. code-block:: python

    frame_transform = transforms.Compose([
        transforms.ToFrame(sensor_size=sensor_size, time_window=10000)
    ])

    train_dataset = tonic.datasets.SHD(save_to='../data', train=True, transform=frame_transform)

The dataloader uses ``PadTensors(batch_first=False)`` so the resulting batch is arranged as a time-major sequence. The
training loop then calls the model one timestep at a time.

Model
-----

The SNN model is intentionally direct:

.. code-block:: python

    class SHD_SNN(tt.Model):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(sensor_size[0], 256),
                tt.snn.LIB(256, beta=torch.rand(256), threshold=torch.rand(256)),
                nn.Linear(256, 256),
                tt.snn.LIB(256, beta=torch.rand(256), threshold=torch.rand(256)),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.net(x)

The GRU and S6 models use the same outer training pattern. This is the core traceTorch idea: the layer dynamics can
change while the model and loop shape stay familiar.

Training loop
-------------

The sequence is processed explicitly:

.. code-block:: python

    model.zero_states()

    for t in range(seq_len):
        output = model(events[t])

    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()

Only the final output is used for the loss in this example. Other tasks may accumulate loss over all timesteps or
average outputs, as the MNIST rate-coded example does.

Notes
-----

The example is meant to demonstrate event-data integration and traceTorch state management. It is not tuned as an SHD
benchmark recipe.
