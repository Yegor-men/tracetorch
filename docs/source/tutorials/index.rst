Tutorials
=========

These tutorials show how to build and train models with traceTorch. We'll work
through three tasks—handwritten digits, auditory events and next-byte
prediction—then extend the image task with a custom neuron. The small workloads
give us time to inspect what happens and try changes; the goal is to understand
the models and their training, rather than produce state-of-the-art results.

You can read each notebook here with its saved figures, or open it and work
through the cells yourself. To get the source, follow :doc:`../installation/index`:
clone the repository, install the chosen example's ``requirements.txt``, and
open its ``tutorial.ipynb``. The adjacent ``main.py`` runs the finished workflow.
The links below open those files on GitHub.

.. list-table::
   :header-rows: 1

   * - Task
     - Notebook
     - Complete program
   * - Image patches
     - `MNIST notebook <https://github.com/Yegor-men/tracetorch/blob/main/examples/mnist/tutorial.ipynb>`_
     - `MNIST program <https://github.com/Yegor-men/tracetorch/blob/main/examples/mnist/main.py>`_
   * - Auditory events
     - `SHD notebook <https://github.com/Yegor-men/tracetorch/blob/main/examples/shd/tutorial.ipynb>`_
     - `SHD program <https://github.com/Yegor-men/tracetorch/blob/main/examples/shd/main.py>`_
   * - Alice language model
     - `Language notebook <https://github.com/Yegor-men/tracetorch/blob/main/examples/alice_language_model/tutorial.ipynb>`_
     - `Language program <https://github.com/Yegor-men/tracetorch/blob/main/examples/alice_language_model/main.py>`_
   * - Custom neuron
     - `Custom-neuron notebook <https://github.com/Yegor-men/tracetorch/blob/main/examples/custom_neuron/tutorial.ipynb>`_
     - `Custom-neuron program <https://github.com/Yegor-men/tracetorch/blob/main/examples/custom_neuron/main.py>`_

All examples share downloaded data under ``examples/data/``. Each keeps its
figures and ``checkpoints/model.pt`` inside its own folder; where continuing
an unfinished sequence matters, we'll save its memory in
``checkpoints/states.pt`` too.

The order below is a suggested path. MNIST introduces stateful training and
class-evidence accumulation. SHD adds synaptic filtering, varied decays and a
feature accumulator with a learned readout. The language model uses residual
context and one-step online updates. The custom-neuron lesson then explores
state shapes and parameter transformations in more detail.

.. toctree::
   :maxdepth: 1

   mnist
   shd
   language_model
   custom_layer
