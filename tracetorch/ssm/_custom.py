import torch
from torch import nn
from ._ssmlayer import Layer as SSMLayer
from typing import Union, Literal


class SelectiveSSM(SSMLayer):
    def __init__(self, num_neurons: int, d_state: int = 16, decay: Union[float, torch.Tensor] = 0.9,
                 decay_rank: Literal[0, 1] = 1, learn_decay: bool = True, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=d_state)

        self._register_state_scale("scale", decay, decay_rank, learn_decay)

        self.A_proj = nn.Linear(num_neurons, d_state)
        self.B_proj = nn.Linear(num_neurons, num_neurons)
        self.C = nn.Parameter(torch.randn(d_state))
        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        delta = nn.functional.softplus(self.A_proj(x_w))  # [..., N]
        decay = torch.exp(delta * -self.scale).unsqueeze(-2)  # [..., 1, N]
        b_x = self.B_proj(x_w)  # [..., D]

        state = self._state_to_working_dim(self.state)  # [..., D, N]
        state = state * decay + b_x.unsqueeze(-1) * (1 - decay)
        self.state = self._state_from_working_dim(state)

        y = torch.sum(state * self.C, dim=-1)
        return self._from_working_dim(y)


class SelectiveZOHSSM(SSMLayer):
    def __init__(self, num_neurons: int, d_state: int = 16, decay: Union[float, torch.Tensor] = 0.9,
                 decay_rank: Literal[0, 1] = 1, learn_decay: bool = True, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=d_state)

        self._register_log_state_scale("scale", decay, decay_rank, learn_decay)

        self.A_proj = nn.Linear(num_neurons, d_state)
        self.B_proj = nn.Linear(num_neurons, num_neurons)
        self.C = nn.Parameter(torch.randn(d_state))
        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        delta = nn.functional.softplus(self.A_proj(x_w))
        A_eff = -self.scale

        delta_A = delta * A_eff
        bar_A = torch.exp(delta_A).unsqueeze(-2)  # [..., 1, N]
        bar_B = ((torch.exp(delta_A) - 1) / (A_eff + 1e-12) * delta).unsqueeze(-2)  # [..., 1, N]

        b_x = self.B_proj(x_w)

        state = self._state_to_working_dim(self.state)
        state = state * bar_A + bar_B * b_x.unsqueeze(-1)
        self.state = self._state_from_working_dim(state)

        y = torch.sum(state * self.C, dim=-1)
        return self._from_working_dim(y)


class SpikeSSM(SSMLayer):
    """SNN Layer serves as the state. B projects D to N, C projects spikes N to D."""

    def __init__(self, num_neurons: int, lif_layer, d_state: int = 16, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=1)

        self.lif = lif_layer
        self.B = nn.Linear(num_neurons, d_state)
        self.C = nn.Linear(d_state, num_neurons)

    def forward(self, x):
        self._ensure_states(x)  # Ensures base layer stuff if needed, though LIF handles itself
        x_w = self._to_working_dim(x)

        snn_in = self.B(x_w)
        spikes = self.lif(snn_in)
        y = self.C(spikes)

        return self._from_working_dim(y)
