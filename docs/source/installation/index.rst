Installation
============

traceTorch is available on PyPI and can be installed via pip.

Basic Installation
------------------

To install the latest stable version of traceTorch, run the following command:

.. code-block:: bash

   pip install tracetorch

Developer Installation
----------------------

If you want to modify traceTorch or run the provided examples, it is recommended to clone the repository and install it in editable mode. This allows you to run the library code locally while experimenting.

.. code-block:: bash

   git clone https://github.com/Yegor-men/tracetorch.git
   cd tracetorch
   pip install -e .

Running Examples
----------------

The ``examples/`` directory contains multiple subprojects (e.g., ``mnist``) showcasing different traceTorch capabilities. Each example project includes its own ``requirements.txt`` for specific dependencies (like ``torchvision`` or ``tqdm``). 

To run the examples:

.. code-block:: bash

   cd examples/mnist
   pip install -r requirements.txt
   python rate_coded.py