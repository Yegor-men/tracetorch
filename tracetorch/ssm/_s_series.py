import torch
from torch import nn
from ._ssmlayer import Layer as SSMLayer


class S4(SSMLayer):
    r"""A diagonal S4-style state-space layer adapted to traceTorch.

    ``S4`` stores a per-feature latent state of size ``d_state`` and updates it
    one timestep at a time. It is designed for traceTorch-style composition, not
    as an optimized replacement for sequence-parallel S4 implementations.

    Args:
        num_neurons (int): number of features in the target dimension.
        d_state (int, default=64): latent state size per feature.
        dim (int, default=-1): dimension along which the layer operates.

    Attributes:
        state: per-feature latent SSM state.
        A_log: log-parameterized diagonal dynamics.
        B: input projection into the state.
        C: output projection from the state.
        D: skip connection scale.
        log_dt: log timestep scale.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where
          ``num_neurons`` is at index ``dim``.
        - **Output**: tensor with the same shape as the input.
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
    r"""An S5-style state-space layer with a global latent state.

    ``S5`` projects the input features into a shared latent state of size
    ``d_state`` and projects that state back to ``num_neurons`` outputs. It
    processes one timestep per forward call and keeps the global state internal.

    Args:
        num_neurons (int): number of features in the target dimension.
        d_state (int, default=64): size of the shared latent state.
        dim (int, default=-1): dimension along which the layer operates.

    Attributes:
        global_state: shared latent state.
        A_log: log-parameterized diagonal dynamics.
        B: input projection into the global state.
        C: output projection from the global state.
        D: skip connection scale.
        log_dt: log timestep scale.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where
          ``num_neurons`` is at index ``dim``.
        - **Output**: tensor with the same shape as the input.
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
    r"""A data-dependent S6 state-space layer adapted to traceTorch.

    ``S6`` is the selective SSM core associated with Mamba-style models, without
    the causal convolution and multiplicative block gate. The timestep, input,
    and output projections are computed from the current input, then applied to
    an internal per-feature state.

    Args:
        num_neurons (int): number of features in the target dimension.
        d_state (int, default=16): latent state size per feature.
        dt_rank (int, default=-1): rank of the timestep projection. ``-1`` uses
            ``max(1, num_neurons // 16)``.
        dim (int, default=-1): dimension along which the layer operates.

    Attributes:
        state: per-feature latent SSM state.
        x_proj: input-dependent projection producing timestep, ``B``, and ``C``.
        dt_proj: projection from low-rank timestep features to per-feature
            timesteps.
        A_log: log-parameterized diagonal dynamics.
        D: skip connection scale.

    Notes:
        - **Input**: tensor of shape ``[*,num_neurons,*]`` where
          ``num_neurons`` is at index ``dim``.
        - **Output**: tensor with the same shape as the input.
    """

    def __init__(self, num_neurons: int, d_state: int = 16, dt_rank: int = -1, dim: int = -1):
        super().__init__(num_neurons, dim, d_state=d_state)
        if dt_rank == -1: dt_rank = max(1, num_neurons // 16)

        self.x_proj = nn.Linear(num_neurons, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, num_neurons, bias=True)
        nn.init.constant_(self.dt_proj.bias, -2.0)

        A = torch.arange(1, d_state + 1).float().repeat(num_neurons, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(num_neurons))
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

        y = torch.sum(state * C.unsqueeze(-2), dim=-1) + x_w * self.D
        return self._from_working_dim(y)


class Mamba(SSMLayer):
    r"""A compact Mamba-style block adapted to traceTorch.

    ``Mamba`` combines an input projection, optional causal convolution buffer,
    SiLU gating, an S6-style selective SSM core, output projection, and residual
    connection. It keeps the convolution buffer and SSM state internal and
    processes one timestep per forward call.

    Args:
        num_neurons (int): number of features in the target dimension.
        d_state (int, default=16): latent SSM state size per feature.
        dim (int, default=-1): dimension along which the layer operates.
        dt_rank (int, default=-1): rank of the timestep projection. ``-1`` uses
            ``max(1, num_neurons // 16)``.
        conv_kernel (int, default=4): causal convolution buffer length. Values
            ``<= 1`` disable the convolution buffer.

    Attributes:
        ssm_state: per-feature selective SSM state.
        conv_buffer: causal convolution buffer, present when ``conv_kernel > 1``.

    Notes:
        This is a traceTorch-compatible experimental implementation. It is not
        an optimized replacement for production Mamba kernels.
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
