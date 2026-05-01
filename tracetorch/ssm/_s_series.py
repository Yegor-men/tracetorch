import torch
from torch import nn
from ._ssmlayer import Layer as SSMLayer


class S4(SSMLayer):
    """
    S4 (Structured State Space Sequence) Layer adapted for traceTorch.
    
    Args:
        num_neurons (int): The number of features in the working dimension.
        d_state (int): The dimension of the state space.
        dim (int): The dimension to operate on (default: -1).
    """

    def __init__(self, num_neurons: int, d_state: int = 64, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=d_state)
        A = torch.arange(1, d_state + 1).float().repeat(num_neurons, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.B = nn.Parameter(torch.randn(num_neurons, d_state))
        self.C = nn.Parameter(torch.randn(num_neurons, d_state))
        self.log_dt = nn.Parameter(torch.randn(num_neurons))
        self.D = nn.Parameter(torch.randn(num_neurons))
        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A = -torch.exp(self.A_log)

        bar_A = torch.exp(dt * A)
        bar_B = (bar_A - 1) / A * self.B

        state = self._state_to_working_dim(self.state)
        state = state * bar_A + bar_B * x_w.unsqueeze(-1)
        self.state = self._state_from_working_dim(state)

        y = torch.sum(state * self.C, dim=-1) + x_w * self.D
        return self._from_working_dim(y)


class S5(SSMLayer):
    """
    S5 (Simplified State Space) Layer adapted for traceTorch.
    Uses a global state shared across neurons.
    
    Args:
        num_neurons (int): The number of features in the working dimension.
        d_state (int): The dimension of the global state space.
        dim (int): The dimension to operate on (default: -1).
    """

    def __init__(self, num_neurons: int, d_state: int = 64, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=1)
        self.d_state = d_state
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.B = nn.Parameter(torch.randn(num_neurons, d_state))
        self.C = nn.Parameter(torch.randn(d_state, num_neurons))
        self.D = nn.Parameter(torch.randn(num_neurons))
        self.log_dt = nn.Parameter(torch.randn(1))
        self._initialize_state("global_state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        dt = torch.exp(self.log_dt)
        A = -torch.exp(self.A_log)

        bar_A = torch.exp(dt * A)
        bar_B = (bar_A - 1) / A

        g_state = self._to_working_dim(self.global_state)
        x_b = torch.matmul(x_w, self.B)
        g_state = g_state * bar_A + bar_B * x_b
        self.global_state = self._from_working_dim(g_state)

        y = torch.matmul(g_state, self.C) + x_w * self.D
        return self._from_working_dim(y)

    def _ensure_state(self, state_name: str, reference_tensor: torch.Tensor):
        state = getattr(self, state_name)
        if state is None:
            shape = list(reference_tensor.shape)
            shape[self.dim] = self.d_state
            state = torch.zeros(shape, dtype=reference_tensor.dtype, device=reference_tensor.device)
            setattr(self, state_name, state)


class S6(SSMLayer):
    """S6 (Data-Dependent State Space) Layer adapted for traceTorch.
    The core dynamic component of Mamba without the causal convolution and multiplicative gating.

    Args:
        num_neurons (int): The number of features in the working dimension.
        d_state (int): The dimension of the state space.
        dt_rank (int): Rank of the step size projection.
        dim (int): The dimension to operate on (default: -1).
    """

    def __init__(self, num_neurons: int, d_state: int = 16, dt_rank: int = -1, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=d_state)
        if dt_rank == -1: dt_rank = max(1, num_neurons // 16)

        self.x_proj = nn.Linear(num_neurons, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, num_neurons, bias=True)
        nn.init.constant_(self.dt_proj.bias, -2.0)

        A = torch.arange(1, d_state + 1).float().repeat(num_neurons, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(num_neurons))  # <--- Fixed: Added missing skip connection
        self._initialize_state("state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        x_proj = self.x_proj(x_w)
        dt_raw, B, C = torch.split(x_proj, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)

        dt = nn.functional.softplus(self.dt_proj(dt_raw)).unsqueeze(-1)
        A = -torch.exp(self.A_log)
        B = B.unsqueeze(-2)

        bar_A = torch.exp(dt * A)
        bar_B = (bar_A - 1) / (A + 1e-12) * B

        state = self._state_to_working_dim(self.state)
        state = state * bar_A + bar_B * x_w.unsqueeze(-1)
        self.state = self._state_from_working_dim(state)

        y = torch.sum(state * C.unsqueeze(-2), dim=-1) + x_w * self.D  # <--- Fixed: Added + x_w * self.D
        return self._from_working_dim(y)


class Mamba(SSMLayer):
    """
    The full Mamba block natively inlined.
    Integrates a Causal Conv1D, SiLU Multiplicative Gating, the S6 Data-Dependent SSM core,
    and a Block-Level Residual Connection.
    """

    def __init__(self, num_neurons: int, d_state: int = 16, dim: int = -1, dt_rank: int = -1, conv_kernel: int = 4):
        # We pass d_state to super() so the base _ensure_state handles our ssm_state natively
        super().__init__(num_neurons, dim, d_state=d_state)

        self.conv_kernel = conv_kernel

        # 1. Outer Block Projections
        self.in_proj = nn.Linear(num_neurons, num_neurons * 2, bias=False)
        self.out_proj = nn.Linear(num_neurons, num_neurons, bias=False)

        # 2. Causal Conv1D Buffer
        if conv_kernel > 1:
            self.conv_weights = nn.Parameter(torch.ones(num_neurons, conv_kernel) / conv_kernel)
            self.conv_bias = nn.Parameter(torch.zeros(num_neurons))
            self._initialize_state("conv_buffer")

        # 3. Inner S6 (SSM) Core Parameters
        if dt_rank == -1:
            dt_rank = max(1, num_neurons // 16)

        self.ssm_proj = nn.Linear(num_neurons, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, num_neurons, bias=True)
        nn.init.constant_(self.dt_proj.bias, -2.0)  # Prevents exploding dt on step 1

        A = torch.arange(1, d_state + 1).float().repeat(num_neurons, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(num_neurons))  # SSM Skip Connection
        self._initialize_state("ssm_state")

    def forward(self, x):
        self._ensure_states(x)
        x_w = self._to_working_dim(x)

        # 1. Project and split into Main branch and Gate branch
        x_proj = self.in_proj(x_w)
        x_main, x_gate = x_proj.chunk(2, dim=-1)

        # 2. Causal 1D Convolution over the time window
        if self.conv_kernel > 1:
            # We use _state_to_working_dim because conv_buffer has the kernel appended at the end!
            conv_buf = self._state_to_working_dim(self.conv_buffer)
            conv_buf = torch.cat([conv_buf[..., 1:], x_main.unsqueeze(-1)], dim=-1)
            self.conv_buffer = self._state_from_working_dim(conv_buf)

            x_conv = torch.sum(conv_buf * self.conv_weights, dim=-1) + self.conv_bias
        else:
            x_conv = x_main

        x_conv = nn.functional.silu(x_conv)

        # 3. Native Data-Dependent SSM (S6 Math)
        ssm_proj_out = self.ssm_proj(x_conv)
        dt_raw, B, C = torch.split(ssm_proj_out, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)

        dt = nn.functional.softplus(self.dt_proj(dt_raw)).unsqueeze(-1)
        A = -torch.exp(self.A_log)
        B = B.unsqueeze(-2)

        bar_A = torch.exp(dt * A)
        bar_B = (bar_A - 1) / (A + 1e-12) * B

        state = self._state_to_working_dim(self.ssm_state)
        state = state * bar_A + bar_B * x_conv.unsqueeze(-1)
        self.ssm_state = self._state_from_working_dim(state)

        y_ssm = torch.sum(state * C.unsqueeze(-2), dim=-1) + x_conv * self.D

        # 4. Multiplicative Gating branch
        y = y_ssm * nn.functional.silu(x_gate)
        y = self.out_proj(y)

        # 5. Block-Level Residual Connection
        return self._from_working_dim(y + x_w)

    def _ensure_state(self, state_name: str, reference_tensor: torch.Tensor):
        if state_name == "conv_buffer":
            state = getattr(self, state_name)
            if state is None:
                shape = list(reference_tensor.shape)
                shape[self.dim] = self.num_neurons
                shape.append(self.conv_kernel)
                state = torch.zeros(shape, dtype=reference_tensor.dtype, device=reference_tensor.device)
                setattr(self, state_name, state)
        else:
            # Passes ssm_state to the base layer, which handles self.d_state cleanly
            super()._ensure_state(state_name, reference_tensor)
