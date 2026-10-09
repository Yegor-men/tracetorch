from ..core import Layer as BaseLayer


class Layer(BaseLayer):
    """Base for RNN layers, inheriting state management from ``tt.Layer``.

    Built-in RNN layers process one timestep with features on the last axis.
    """
