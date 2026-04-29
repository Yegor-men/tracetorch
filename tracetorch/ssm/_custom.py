import torch
from torch import nn
from ._ssmlayer import Layer as SSMLayer
from typing import Union, Literal


class SelectiveSSM(SSMLayer):
    def __init__(
            self,
            num_neurons: int,
            d_state: int = 16,
            decay: Union[float, torch.Tensor] = 0.9,
            dim: int = -1,
            decay_rank: Literal[0, 1] = 1,
            learn_decay: bool = True,
    ):
        super().__init__(num_neurons, dim, d_state=d_state)

        self._register_state_scale("scale", decay, decay_rank, learn_decay)

        self.ABD = nn.Linear(num_neurons, d_state + num_neurons + num_neurons)
        nn.init.zeros_(self.ABD.bias)

        self.C = nn.Linear(d_state, 1)
        nn.init.zeros_(self.C.bias)

        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        A, B, D = torch.split(self.ABD(x_w), [self.d_state, self.num_neurons, self.num_neurons], dim=-1)
        # A, B, D are [..., D], [..., N], [..., D]

        delta = nn.functional.softplus(A)  # [..., D]
        decay = torch.exp(delta * -self.scale).unsqueeze(-2)  # [..., 1, D]

        state = self._state_to_working_dim(self.state)  # [..., N, D]
        state = state * decay + B.unsqueeze(-1) * (1 - decay)
        self.state = self._state_from_working_dim(state)

        out = x_w + self.C(state).squeeze(-1) * nn.functional.silu(D)
        out = self._from_working_dim(out)

        return out


class SelectiveZOHSSM(SSMLayer):
    def __init__(
            self,
            num_neurons: int,
            d_state: int = 16,
            dim: int = -1,
            learn_decay: bool = True
    ):
        super().__init__(num_neurons, dim, d_state=d_state)

        self._register_log_state_scale("scale", torch.arange(1, d_state + 1), 1, learn_decay)

        self.ABD = nn.Linear(num_neurons, d_state + num_neurons + num_neurons)
        nn.init.zeros_(self.ABD.bias)

        self.C = nn.Linear(d_state, 1)
        nn.init.zeros_(self.C.bias)

        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        A, B, D = torch.split(self.ABD(x_w), [self.d_state, self.num_neurons, self.num_neurons], dim=-1)
        # A, B, D are [..., D], [..., N], [..., D]

        delta = nn.functional.softplus(A)
        delta_A = delta * -self.scale

        bar_A = torch.exp(delta_A).unsqueeze(-2)  # [..., 1, D]
        bar_B = ((torch.exp(delta_A) - 1) / (-self.scale + 1e-12) * delta).unsqueeze(-2)  # [..., 1, D]

        state = self._state_to_working_dim(self.state)
        state = state * bar_A + bar_B * B.unsqueeze(-1)
        self.state = self._state_from_working_dim(state)

        out = x_w + self.C(state).squeeze(-1) * nn.functional.silu(D)
        out = self._from_working_dim(out)

        return out


class SelectiveSNN(SSMLayer):
    def __init__(self, num_neurons: int, snn_layer, dim: int = -1):
        super().__init__(num_neurons, dim, snn_layer.num_neurons)

        self.snn_layer = snn_layer
        self.B = nn.Linear(num_neurons, num_neurons, bias=False)
        self.C = nn.Linear(self.d_state, 1)
        self.D = nn.Linear(num_neurons, num_neurons)
        nn.init.zeros_(self.C.bias)
        # nn.init.zeros_(self.D.weight)

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        snn_in = self.B(x_w)
        snn_in_expanded = snn_in.unsqueeze(-1).expand(*snn_in.shape, self.d_state)
        spikes = self.snn_layer(snn_in_expanded)
        y = self.C(spikes).squeeze(-1)

        D = self.D(x_w)
        out = x_w + y * nn.functional.silu(D)

        return self._from_working_dim(out)
