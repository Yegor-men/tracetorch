from .core import Layer as Layer
from .core import Model as Model
import importlib

from . import core, functional, rnn, snn

__all__ = [
    "Layer",
    "Model",
    "core",
    "functional",
    "plot",
    "rnn",
    "snn",
]


def __getattr__(name):
    """Lazily import optional top-level submodules."""
    if name == "plot":
        plot = importlib.import_module(f"{__name__}.plot")
        globals()["plot"] = plot
        return plot
    raise AttributeError(f"module 'tracetorch' has no attribute {name!r}")
