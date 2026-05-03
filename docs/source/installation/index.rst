Installation
============

Basic Installation
------------------
traceTorch is available on PyPI and can be installed via pip:

.. code-block:: bash

    pip install tracetorch

traceTorch depends on ``torch>=2.0.0``, ``numpy>=1.20.0``, ``matplotlib>=3.0.0``, ``scipy>=1.10.0``, which will be automatically installed.

Developer Installation
----------------------
If you want to modify traceTorch or run the examples without copy pasting the code, it is recommended to clone the repository
and install it in editable mode.

.. code-block:: bash

    git clone https://github.com/Yegor-men/tracetorch.git
    cd tracetorch
    pip install -e .

The ``examples/`` directory contains multiple subprojects (e.g., ``mnist/``) showcasing different traceTorch capabilities.
Each example project includes its own ``requirements.txt`` for specific dependencies (like ``torchvision`` or ``tqdm``).

To run the examples:

.. code-block:: bash

    cd examples/mnist
    pip install -r requirements.txt
    python rate_coded.py