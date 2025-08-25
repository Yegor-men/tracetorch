import torch
from torch import nn


class LIS(nn.Module):
    def __init__(
            self,
            num_in: int,
            num_out: int,
            pre_decay: float = 0.9,
    ):
        super().__init__()
        self.num_in = int(num_in)
        self.num_out = int(num_out)

        # parameters: weight [out, in], bias [out], pre_decay raw [in]
        # keep consistency with ALIF: store decay in raw space (use sigmoid_inverse if you want)
        # For simplicity here we store raw value that will be pushed through sigmoid when used.
        self.params = nn.ParameterDict({
            "weight": nn.Parameter(torch.randn(num_out, num_in)),
            "bias": nn.Parameter(torch.zeros(num_out)),
            # store pre_decay as raw param so we can apply sigmoid(raw) in forward/backward
            "pre_decay": nn.Parameter(torch.full((num_in,), pre_decay))
        })

        # per-parameter eligibility accumulators (store real grads, already multiplied by learning signal)
        for name, p in self.params.items():
            self.register_buffer(f"{name}_elig", torch.zeros_like(p), persistent=True)

        # running traces / snapshots
        self.register_buffer("pre_trace", torch.zeros(num_in))
        self.register_buffer("normalized_pre_trace", torch.zeros(num_in))

        self.register_buffer("last_pre_trace", torch.zeros(num_in))
        # recurrence trace used for pre_decay eligibility: tracks ∂pre_trace/∂pre_decay over time
        self.register_buffer("pre_decay_trace", torch.zeros(num_in))

    @torch.no_grad()
    def zero_states(self):
        # reset runtime traces
        self.pre_trace.zero_()
        self.normalized_pre_trace.zero_()
        self.last_pre_trace.zero_()
        self.pre_decay_trace.zero_()

    @torch.no_grad()
    def zero_elig(self):
        # reset elig accumulators (real gradient accumulators)
        for name in self.params.keys():
            elig = getattr(self, f"{name}_elig")
            elig.zero_()

    @torch.no_grad()
    def forward(self, in_spikes):
        """
        in_spikes: [in]
        returns softmax probability distribution: [out]
        """
        # decode actual pre_decay
        pre_decay = torch.sigmoid(self.params["pre_decay"])  # [in]

        # snapshot previous pre_trace (used in recurrence)
        self.last_pre_trace.zero_().add_(self.pre_trace)

        # update pre-trace (running presynaptic trace)
        self.pre_trace.mul_(pre_decay).add_(in_spikes)

        # normalized pre used for weight grads (average-style)
        self.normalized_pre_trace.zero_().add_(self.pre_trace * (1.0 - pre_decay))

        # compute logits and probabilities
        logits = torch.einsum("oi,i->o", self.params["weight"], self.normalized_pre_trace) + self.params["bias"]
        probability_distribution = nn.functional.softmax(logits, dim=-1)

        # update recurrence trace for pre_decay eligibility:
        # e_pre(t) = last_pre_trace + pre_decay * e_pre(t-1)
        self.pre_decay_trace.mul_(pre_decay).add_(self.last_pre_trace)

        return probability_distribution

    @torch.no_grad()
    def backward(self, learning_signal):
        """
        learning_signal: shape [out]  (dL/dlogit or equivalent)
        This accumulates real gradients into *_elig buffers.
        Returns: passed learning signal upstream: shape [in]
        """

        # decode actual pre_decay for normalization
        pre_decay = torch.sigmoid(self.params["pre_decay"])  # [in]

        # normalized_pre is already stored in self.normalized_pre_trace
        normalized_pre = self.normalized_pre_trace  # [in]

        # ---------- weight eligibility ----------
        # real gradient per-weight at this timestep = outer(learning_signal, normalized_pre)
        # shape [out, in]
        weight_e_inc = torch.einsum("o,i->oi", learning_signal, normalized_pre)
        self.weight_elig.add_(weight_e_inc)  # accumulate real grad into weight_elig

        # ---------- bias eligibility ----------
        # real gradient for bias is just the learning signal
        self.bias_elig.add_(learning_signal)  # [out]

        # ---------- pre_decay eligibility ----------
        # For each input i, compute S_i = sum_j learning_signal_j * weight_j,i
        # This is how much the outputs depend on that input's normalized_pre.
        # Then the contribution to pre_decay's gradient is pre_decay_trace[i] * S_i
        S = torch.einsum("o,oi->i", learning_signal, self.params["weight"])  # [in]
        self.pre_decay_elig.add_(self.pre_decay_trace * S)  # [in]

        # ---------- passed upstream learning signal ----------
        # pass through weight^T and scale by (1 - pre_decay) because normalized_pre = pre_trace * (1 - pre_decay)
        passed_ls = torch.einsum("oi,o->i", self.params["weight"], learning_signal) * (1.0 - pre_decay)

        return passed_ls

    @torch.no_grad()
    def elig_to_grad(self, scalar: float = 1.0):
        """
        Move accumulated eligibilities (real grads w.r.t actual param values)
        into raw parameter .grad (applying transform derivatives where needed).
        """
        def accum_grad(param, g_raw):
            if param.grad is None:
                param.grad = g_raw.clone()
            else:
                param.grad.add_(g_raw)

        # --- weight (identity transform) ---
        w = self.params["weight"]
        if getattr(self, "weight_elig", None) is not None:
            accum_grad(w, scalar * self.weight_elig)
            self.weight_elig.zero_()

        # --- bias (identity) ---
        b = self.params["bias"]
        if getattr(self, "bias_elig", None) is not None:
            accum_grad(b, scalar * self.bias_elig)
            self.bias_elig.zero_()

        # --- pre_decay (actual = sigmoid(raw)): raw_grad = actual_grad * s * (1 - s) ---
        raw = self.params["pre_decay"]
        s = torch.sigmoid(raw)
        g_raw = scalar * (self.pre_decay_elig * (s * (1.0 - s)))
        accum_grad(raw, g_raw)
        self.pre_decay_elig.zero_()
