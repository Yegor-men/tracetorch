![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-purple.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![PyPI](https://img.shields.io/badge/PyPI-v0.2.0-blue.svg)](https://pypi.org/project/tracetorch/)

``traceTorch`` is a PyTorch-based library built on the principles of spiking neural networks, replacing the PyTorch
default backpropagation through time with lightweight, per-layer input traces, enabling biologically inspired, constant
time and memory consumption learning on arbitrarily long or even streaming sequences.

## Documentation

It is highly recommended that you read the [documentation](https://yegor-men.github.io/tracetorch/) first. It contains:

1. **Introduction**: An introduction to traceTorch, how and why it works, it's founding principles. It's thoroughly
   recommended that you read through the entire introduction and gain an intuitive understanding before proceeding.
2. **Tutorials**: Various tutorials to create your own traceTorch models. The resultant code can be found in
   `tutorials/`.
3. **Documentation**: The actual documentation to all the modules included in `traceTorch`. It includes detailed
   explanations, examples and math to gain a full understanding.

## Roadmap

- Create the poisson click test example
- Implement the trace alternative to REINFORCE
- Finish writing the documentation
- Move tutorial code to separate repository
- Implement abstract graph based models, not just sequential

## Installation

``traceTorch`` is a PyPI library, which can be found [here](https://pypi.org/project/tracetorch/).

You can install it via pip. All the required packages for it to work are also downloaded automatically.

```
pip install tracetorch
```

To use, simply do:

```
import tracetorch
```

## Usage examples

`tutorials/` contains all the tutorial files, ready to run and playtest. The tutorials themselves can be found
[here](https://yegor-men.github.io/tracetorch/tutorials/index.html).

The tutorials make use of libraries that ``tracetorch`` doesn't necessarily use. To ensure that you have all the
necessary packages for the tutorials installed, please install the packages listed in `tutorials/requirements.txt`

```
cd tutorials/
pip install -r requirements.txt
```

It's recommended to use an environment that does _not_ have ``tracetorch`` installed if using the tutorials,
``tracetorch/`` is structured identically to the library, but is of course a running release.

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Acknowledgements

I built traceTorch from the ground up, trying to reverse engineer biological neurons with a sprinkle of intelligent
design, but I would also like to recognize the following projects and people who helped shape my thinking:

- [snntorch](https://github.com/jeshraghian/snntorch) for introducing me to SNN networks in the first place, and their
  principles of function. Ironically, its dependency on constructing the full autograd graph is what largely inspired me
  to make ``traceTorch``.
- [Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov) for introducing me to computational neuroscience, presenting
  interesting concepts in an easy-to-understand manner. My earliest tests, when I naively wanted to implement 1:1
  biological neurons, largely revolved around his work.
- [e-prop (eligibility propagation)](https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf) inspired
  the whole "trace" concept, the idea of keeping a decaying value. Earlier, before ``traceTorch``, I wanted to use
  e-prop for online learning instead. Admittedly unsuccessful in my attempts, and a little put off by the relative
  difficulty, I instead wanted to make something simpler.

## Contributing

Contributions are always welcome. Feel free to submit pull requests or report issues, I will occasionally check in on
it.

You can also reach out to me via either email or Twitter:

- yegor.mn@gmail.com
- [Twitter](https://x.com/Yegor_Men)
