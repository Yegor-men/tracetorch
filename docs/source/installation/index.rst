Installation
============

We can install traceTorch into a Python environment with pip:

.. code-block:: bash

   pip install tracetorch

The library depends on PyTorch, so we'll need a PyTorch installation that supports
the CPU or GPU we plan to use. traceTorch requires Python 3.8 or newer, although
the selected PyTorch release may require a newer version.

Getting the examples
--------------------

The runnable examples live in the repository, with a notebook beside each program
when a tutorial is available. To get that source and install the downloaded
version of the library:

.. code-block:: bash

   git clone https://github.com/Yegor-men/tracetorch.git
   cd tracetorch
   pip install -e .

We can now run :doc:`../quickstart` without installing anything else or downloading
a dataset:

.. code-block:: bash

   python examples/quickstart/main.py

The quickstart trains a small model to remember an initial value through several
timesteps. Its complete loop gives us a starting point before the background
material and practical tutorials.

Running a practical example
---------------------------

Each practical example keeps its additional packages in a local
``requirements.txt``. We'll install those for the task we want to try, then run
its ``main.py``. For MNIST, using the terminal in the downloaded repository:

.. code-block:: bash

   pip install -r examples/mnist/requirements.txt
   python examples/mnist/main.py --device cpu

The program trains the model and saves figures and a checkpoint inside
``examples/mnist``. Downloaded data goes into the shared ``examples/data`` folder,
so another example using MNIST can reuse it. Other example folders follow the
same pattern, with their own requirements and main files.

Working through a notebook
--------------------------

The same requirements include JupyterLab. Once they're installed, we can open
the interactive version:

.. code-block:: bash

   jupyter lab

Open ``examples/mnist/tutorial.ipynb`` and run its cells in order. The notebook
builds the solution step by step, so we can inspect intermediate values, change
a setting and rerun an experiment. An IDE with Jupyter support can open it too;
use the environment where the example's packages are installed.

The web :doc:`../tutorials/index` display these same notebooks with their saved
figures and results. ``main.py`` is the complete automatic version, while
``tutorial.ipynb`` provides the explanation and room to experiment.

Working on the documentation
----------------------------

The `hosted documentation <https://yegor-men.github.io/tracetorch/>`_ is the main
reading version and updates automatically after a push to the main branch.
If you're changing the documentation, you can also build it locally:

.. code-block:: bash

   pip install -r docs/requirements.txt
   cd docs/
   make html

The build uses saved notebook outputs, so it doesn't run training or download
datasets. The requirements include Pandoc for notebook rendering, and the
generated pages appear under ``docs/build/html``.

The small ``.nblink`` files in ``docs/source/tutorials`` point Sphinx to the
notebooks under ``examples/``. They let the website render each notebook,
including its saved outputs, without keeping another copy of the lesson.
