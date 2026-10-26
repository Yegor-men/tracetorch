# Examples and their tutorials

Each folder contains one example, with `main.py` as the program that trains and saves the model. When a tutorial is available, `tutorial.ipynb` sits beside it and builds the same solution through explanations, code cells, figures and experiments. The folder's `requirements.txt` contains the additional packages for both versions.

The practical tasks show how to build and train models with traceTorch. Their small training budgets leave room to inspect results and try changes; they are not intended to produce state-of-the-art models. The [online tutorials](https://yegor-men.github.io/tracetorch/tutorials/index.html) show the notebooks with saved results, so reading them needs no local documentation build.

After cloning the repository and installing it with `pip install -e .`, you can run the small, download-free quickstart:

```bash
python examples/quickstart/main.py
```

For a practical example, install its requirements and run its main file. For MNIST that is:

```bash
pip install -r examples/mnist/requirements.txt
python examples/mnist/main.py --device cpu
```

Those requirements also install JupyterLab, so you can run `jupyter lab` and open `examples/mnist/tutorial.ipynb` if you'd like to work through the lesson. Each notebook contains its own explanation and model code; it doesn't require running the complete program first.

| Folder | What we'll explore |
| --- | --- |
| `quickstart/` | Remembering an initial random value through later zero inputs; only the library's runtime packages are needed |
| `mnist/` | Reading 49 image patches, training from the final output and inspecting predictions |
| `shd/` | Filtering auditory events, retaining features at different decays and using compiled evaluation |
| `alice_language_model/` | Predicting bytes with a residual SNN, updating every timestep and restoring an unfinished stream |
| `custom_neuron/` | Declaring matrix-shaped synaptic histories and learning how to mix them |

The folder names don't impose a lesson order. The documentation suggests a reading path, and an example can have additional supporting files as it grows; `main.py` remains the entry point. A folder without `tutorial.ipynb` simply has no accompanying walkthrough.

## Data and saved results

All examples share downloaded data under `examples/data/`. In particular, MNIST and the custom-neuron example reuse the same images. Each example keeps its own generated results, for example:

```text
examples/
├── data/
└── alice_language_model/
    ├── main.py
    ├── tutorial.ipynb
    ├── requirements.txt
    ├── figures/
    └── checkpoints/
        ├── model.pt
        └── states.pt
```

`model.pt` holds learned weights, optimizer state and the configuration needed to reconstruct the model. The continuation examples also write `states.pt`, which holds hidden-state snapshots and the position of the next input. Independent image and recording tasks start from reset states, so they only need the model checkpoint. Running an example again replaces its default result files; you can copy a run's folder when you want to retain it.

Notebooks use `../data` and their local `checkpoints` folder when running from their own example folder. The programs resolve these defaults from `main.py`, so their data and output locations don't change with the terminal's working folder.

## Workload choices

The practical programs select CUDA when available; `--device cpu` reproduces the CPU setup used in the tutorials. Use `--help` to see the training and output options. Defaults use small subsets or a bounded amount of text, while zero MNIST/SHD sample limits select the full split.

SHD still downloads its full source dataset before selecting recordings, so its initial download can be much longer than training. The language example preserves Project Gutenberg's source and licence for [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11); `--corpus /path/to/enwik8 --max-bytes 100000` lets you use a local uncompressed corpus instead.
