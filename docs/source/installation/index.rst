Installation
============

PyPI install
------------

Install traceTorch with pip:

.. code-block:: bash

    pip install tracetorch

The package depends on PyTorch, NumPy, Matplotlib, and SciPy. PyTorch installation can be platform-specific, especially
when choosing a CUDA build; if you already have a working PyTorch environment, install traceTorch into that environment.

Editable install
----------------

Use an editable install if you want to run the repository examples, inspect the source, or work on traceTorch itself.

.. code-block:: bash

    git clone https://github.com/Yegor-men/tracetorch.git
    cd tracetorch
    pip install -e .

Example dependencies
--------------------

Examples have their own requirements. Install them from the example directory you want to run.

MNIST examples:

.. code-block:: bash

    cd examples/mnist
    pip install -r requirements.txt
    python rate_coded.py

Heidelberg Digits example:

.. code-block:: bash

    cd examples/heidelberg_digits
    pip install -r requirements.txt
    python main.py

Documentation dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

    cd docs
    pip install -r requirements.txt
    make html

The generated HTML is written to ``docs/build/html``.

Check the install
-----------------

After installation, this should import successfully:

.. code-block:: python

    import tracetorch as tt
    layer = tt.snn.LIB(num_neurons=16)

If the import fails because of an optional plotting dependency, make sure the base requirements were installed into the
same environment as PyTorch.
