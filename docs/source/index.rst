traceTorch Documentation
========================

.. image:: _static/tracetorch_banner.png
    :alt: traceTorch banner image

.. raw:: html

    <div style="text-align: center; margin: 10px 0;">
        <a href="https://yegor-men.github.io/tracetorch/">
            <img src="https://img.shields.io/pypi/v/tracetorch?style=flat&labelColor=555&label=Documentation&color=red" alt="Documentation">
        </a>
        <a href="https://pypi.org/project/tracetorch/">
            <img src="https://img.shields.io/pypi/v/tracetorch?style=flat&labelColor=555&label=PyPI&color=blue" alt="PyPI version">
        </a>
        <a href="https://opensource.org/license/mit">
            <img src="https://img.shields.io/badge/License-MIT-purple.svg?style=flat&labelColor=555" alt="License">
        </a>
        <a href="https://github.com/Yegor-men/tracetorch/stargazers">
            <img src="https://img.shields.io/github/stars/Yegor-men/tracetorch?style=flat&labelColor=555&label=Stars&color=gold" alt="GitHub stars">
        </a>
        <a href="https://github.com/Yegor-men/tracetorch/network/members">
            <img src="https://img.shields.io/github/forks/Yegor-men/tracetorch?style=flat&labelColor=555&label=Forks&color=green" alt="GitHub forks">
        </a>
        <a href="https://github.com/Yegor-men/tracetorch/issues">
            <img src="https://img.shields.io/github/issues/Yegor-men/tracetorch?style=flat&labelColor=555&label=Issues&color=orange" alt="GitHub issues">
        </a>
        <a href="https://pepy.tech/project/tracetorch">
            <img src="https://static.pepy.tech/personalized-badge/tracetorch?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads" alt="PyPI Downloads">
        </a>
    </div>

A strict, ergonomic, and powerful library for Spiking Neural Networks (SNNs), Recurrent Neural Networks (RNNs), and State Space Models (SSMs) in PyTorch.

traceTorch was designed to seamlessly blend in with the rest of the PyTorch ecosystem, in an intuitive but powerful way.
It handles all the hidden state management boilerplate for you, recursively looking for any traceTorch layer, no matter how deep it's buried in the model.
The layers are designed to work in any setting: whether in an MLP or CNN, the same layer can be used anywhere.
In similar vain, traceTorch gives you the flexibility in how you want your parameters: learnable or not, a scalar or vector or a custom tensor.
If that's not enough, the ecosystem is simple enough that you can easily create your own layers that will seamlessly blend in with the rest of traceTorch.

This documentation is written to help you familiarize with traceTorch: its ethos and intricacies. The examples contain ready to run code,
while the tutorials comprehensively cover everything you ought to know. Having finished them all, you should have all
the necessary knowledge to create and train your own networks: zero, detach, save, and load hidden states and compile models for speedup.
You will also have a solid understanding of all the layers that traceTorch presents, and why things are done a certain way.
Extra tutorials exist that explain how the layers are created, so that you can make your own ones that comply with the rest of the traceTorch ecosystem.

Please beware that traceTorch was primarily developed for SNNs, support for RNNs and SSMs was added later and are not the primary focus.
This means that while RNNs seamlessly integrate with SNNs, SSMs are not so much: the traceTorch implementations are not the
official, optimized (parallelized) ones, rather, they are adapted to be interchangeable with other traceTorch layers: thus only one timestep at a time.
If you're looking for a library for SSMs, traceTorch is likely not it. Furthermore, before proceeding, it is recommended
that you have at least some minimal understanding of SNNs. While the tutorials do cover the basics, they focus more on how
traceTorch does it, rather than the fundamentals and history. If you're completely unfamiliar, the
`snnTorch documentation <https://snntorch.readthedocs.io>`_ is a good place to start.
It is a different, unaffiliated project, with a dissimilar implementation, but the theory behind SNNs is largely the same.


:doc:`Installation <installation/index>`
----------------------------------------
To get started with traceTorch, please visit the installation page for instructions on a basic and developer install.

:doc:`Introduction <introduction/index>`
----------------------------------------
The introduction section covers the necessary background information you ought to know before starting. It covers the ethos of
traceTorch, and subsequently how it is structured to comply. It contains background information on SNNs and justifies
the specific implementation choices. Finally, it briefly mentions all the layers available and discusses some tips on the
practices to modelmaking that yield better results.

:doc:`Examples <examples/index>`
--------------------------------
The examples section recreates ready to run code from the examples found in the `official repository <https://github.com/Yegor-men/tracetorch>`_,
with step by step instructions to explain what's going on and why certain choices are made. This section  will help you
get up and running with making working traceTorch models.

:doc:`Tutorials <tutorials/index>`
----------------------------------
The tutorials section is dedicated to the more advanced parts of traceTorch, such as how to make your own layers, how to
save and load hidden states, and how to compile and decompile trained models.

:doc:`Reference <reference/index>`
----------------------------------
The reference section covers in detail all the modules traceTorch offers: layers, functions, and the likes.

.. toctree::
    :maxdepth: 2
    :caption: Documentation Sections:
    :hidden:

    installation/index
    introduction/index
    tutorials/index
    examples/index
    reference/index