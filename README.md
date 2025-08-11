![traceTorch Banner](media/tracetorch_banner.png)

[![License](https://img.shields.io/badge/License-Apache%202.0-purple.svg)](https://www.apache.org/licenses/LICENSE-2.0)

traceTorch is a PyTorch-based library built on spiking neural network architectures that replaces full backpropagation
through time with lightweight, per-layer input traces, enabling biologically inspired, constant-memory learning on long
or streaming sequences.

It is highly recommended that you read the [documentation](https://yegor-men.github.io/tracetorch/) first. It contains a
detailed introduction to how traceTorch works with intuitive explanations, comparisons and analogies to the "classic"
architecture counterparts. It explains what traceTorch can and can't do, when it's better and when it's not. It also
walks through the `examples/` files in a tutorial manner to explain how you can create your own traceTorch networks.

## Roadmap

- Create the poisson click test example
- Implement the trace alternative to REINFORCE
- Make traceTorch into a PyPI library
- Finish writing documentation

## Installation

⚠️ WARNING, traceTorch is _not yet_ a library. For now, you'll just have to clone this repository and use the
`tracetorch/` folder within.

```
git clone https://github.com/Yegor-men/tracetorch
cd tracetorch/
pip install -r requirements.txt
```

Then, within a python file where from where the repository root folder is visible, simply do:

```
from tracetorch import tracetorch
```

## Usage examples

There exists `tracetorch/examples/` within which sit test files for playtesting, aimed to test if the components work.

The example files are ready to go files that demonstrate traceTorch in various scenarios. To make sure that you have all
the necessary libraries do:

```
pip install -r examples-requirements.txt
```

## Authors

- [@Yegor-men](https://github.com/Yegor-men)

## Acknowledgements

I built traceTorch from the ground up, trying to reverse engineer biological neurons with a sprinkle of intelligent
design, but I would also like to recognize the following projects and people who helped shape my thinking:

- [snntorch](https://github.com/jeshraghian/snntorch) for introducing me to SNN networks in the first place, and their
  principles of function. Ironically, its dependency on constructing the full autograd graph is what largely inspired me
  to make traceTorch.
- [Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov) for introducing me to computational neuroscience, presenting
  interesting concepts in an easy to understand manner. My earliest tests, when I naively wanted to implement 1:1
  biological neurons, largely revolved around his work.
- [e-prop (eligibility propagation)](https://www.biorxiv.org/content/biorxiv/early/2020/04/16/738385.full.pdf) inspired
  the whole "trace" concept, the idea of keeping a decaying value. Earlier, before traceTorch, I wanted to use e-prop
  for online learning instead. Admittedly unsuccessful in my attempts, and a little put off by the relative difficulty,
  I instead wanted to make something simpler.

## Contributing

Contributions are always welcome. Feel free to submit pull requests or report issues, I will occasionally check in on
it.

You can also reach out to me via either email or Twitter:

- email: yegor.mn@gmail.com
- [Twitter](https://x.com/Yegor_Men)
