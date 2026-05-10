Examples
========

The examples mirror the runnable scripts in the repository. They are written to make traceTorch mechanics visible, not
to chase benchmark scores.

Run them from an editable install:

.. code-block:: bash

    git clone https://github.com/Yegor-men/tracetorch.git
    cd tracetorch
    pip install -e .

Then install the requirements for the example you want.

Recommended order:

1. :doc:`mnist` explains the three MNIST scripts: rate-coded, sequential, and noisy.
2. :doc:`shd` explains the Heidelberg Digits event-data example.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    mnist
    shd
