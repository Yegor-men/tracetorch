# Working on traceTorch

traceTorch is a small PyTorch library for stateful recurrent layers. Preserve its explicit model of time: layers process one timestep and own their states; callers decide sequence boundaries and loss placement.

## Library contracts

- `tt.Layer()` defines independently shaped states and tensor parameters. `tt.snn.Layer(num_features)` adds scalar/per-neuron initialization and owns the positive feature count.
- Built-in layers use the final dimension for features. Keep image layout changes explicit.
- Hidden states are plain tensor attributes, separate from weight checkpoints. Reset for independent sequences; detach to bound history while preserving numerical memory.
- Compilation caches activated parameters, preserves raw parameter identity and temporarily freezes them. Decompilation restores intended learnability. Input-dependent activations cannot be cached as parameters.
- Document equations and reset timing precisely. Comments should explain decisions, not narrate obvious operations.

## Tutorials and examples

- Each example lives under `examples/<name>/`. Its `main.py` runs the complete workflow, its optional `tutorial.ipynb` teaches the same solution, and its local `requirements.txt` covers both. Do not encode lesson order in file or folder names. Update code, notebooks, navigation and meaningful verification together.
- The documentation renders the notebooks directly. Do not maintain another copy of their narration in documentation pages. Use Sphinx `literalinclude` for the small quickstart and other source references where useful.
- Keep example folder names lowercase. After installation, users run their main files directly, such as `python examples/mnist/main.py`. Do not explain Python module discovery as part of normal setup. All examples share `examples/data/`; figures and checkpoints belong inside the example's own folder. Save the learned model in `checkpoints/model.pt` and stream memory separately in `checkpoints/states.pt` where continuation is relevant.
- Notebooks define the model and training procedure they teach; do not import a completed example to bypass the lesson. Example programs must not depend on another example. Importing a program must not download data, train, plot, set seeds or change thread counts.
- Defaults should finish quickly on CPU. Measure runtime and held-out learning; a successful backward pass alone does not demonstrate learning.
- Keep timestep loops and state operations visible. Avoid a generic trainer, unnecessary architecture comparisons and unmotivated dependencies.
- Optional dataset dependencies belong to the examples. Core tests need no network or datasets.
- Teach through small steps: explain the input, introduce the relevant mathematics, show the code, then interpret the observed result. Put expected results, figures and a short runtime note in the tutorial itself.
- Use conversational technical English with connected paragraphs. Explain the reasons and connect equations, code and observations rather than presenting a string of short instructions. Define terms and symbols, avoid promotional claims, and do not impose artificial vocabulary or sentence-length limits. Tutorial titles describe the activity, such as "Building a neuron...".
- Present decays directly, using the horizon `1 / (1 - decay)` when a duration helps. Tutorial models demonstrate training and library use, rather than pursuing benchmark results. In the language example, initialize each neuron's membrane decay with `torch.rand(num_features)` and detach states after each timestep's update.
- Order the documentation as introduction, installation, quickstart, Understanding traceTorch, and practical tutorials. The quickstart demonstrates a small temporal task without dataset downloads: remembering an initial random value through later zero inputs. Explain that LI provides memory while remaining linear for a fixed decay. Teach saving, hidden-state continuation and compiled evaluation within the practical tasks where they are useful.

## Verification

Use the project's configured Python interpreter. Run `python -m pytest` for library contracts. Keep the core test suite independent of tutorial scripts and optional dataset packages. Build documentation with `python -m sphinx -W --keep-going -b html docs/source docs/build/html`.

Tests run manually. Do not add test runs to push, tag or release workflows or Git hooks. Keep automatic deployment of the hosted documentation.

For changes to example behavior, run the affected documented command on real data and update the corresponding tutorial's results and runtime note. Check tutorial-specific transformations during development rather than adding a second test suite for the lessons. Treat generated docs, caches, downloaded data and checkpoints as build artifacts. Reference figures included in a tutorial are documentation assets; generate them with the documented example and keep their settings available. Preserve unrelated working-tree and staged changes.

Run every notebook from a fresh kernel before publishing its saved outputs, and check that the outputs are present in the saved file rather than only the live kernel view. Keep notebook and script model definitions and final training behavior aligned; exploratory cells can differ from the automatic program. Documentation builds display saved notebook outputs and must not run training or download data.
