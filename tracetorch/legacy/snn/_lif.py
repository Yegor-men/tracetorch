import torch
from torch import nn
import math
from tracetorch.functional import sigmoid_inverse
from tracetorch.functional import softplus_inverse
from tracetorch.functional import sigmoid_derivative


class LIF(nn.Module):
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float = 0.9,
			threshold: float = 1,
			pre_decay: float = 0.9,
			post_decay: float = 0.9,
	):
		super().__init__()
		self.num_in = int(num_in)
		self.num_out = int(num_out)

		t, i = threshold, num_in
		self.params = nn.ParameterDict({
			"weight": nn.Parameter(torch.normal(mean=t / i, std=t / math.sqrt(i), size=(num_out, num_in))),
			"mem_decay": nn.Parameter(sigmoid_inverse(torch.full((num_out,), mem_decay))),
			"threshold": nn.Parameter(softplus_inverse(torch.full((num_out,), threshold))),
			"pre_decay": nn.Parameter(sigmoid_inverse(torch.full((num_in,), pre_decay))),
			"post_decay": nn.Parameter(sigmoid_inverse(torch.full((num_out,), post_decay))),
			"bias": nn.Parameter(torch.zeros(num_out))
		})

		for name, p in self.params.items():
			self.register_buffer(f"{name}_elig", torch.zeros_like(p), persistent=True)

		self.register_buffer("mem", torch.zeros(num_out))
		self.register_buffer("pre_trace", torch.zeros(num_in))
		self.register_buffer("post_trace", torch.zeros(num_out))

		self.register_buffer("last_pre_trace", torch.zeros(num_in))
		self.register_buffer("last_post_trace", torch.zeros(num_out))
		self.register_buffer("last_mem", torch.zeros(num_out))
		self.register_buffer("last_surrogate", torch.zeros(num_out))

		self.register_buffer("pre_decay_trace", torch.zeros(num_in))
		self.register_buffer("post_decay_trace", torch.zeros(num_out))
		self.register_buffer("mem_decay_trace", torch.zeros(num_out))

	@torch.no_grad()
	def zero_states(self):
		self.mem.zero_()
		self.pre_trace.zero_()
		self.post_trace.zero_()
		self.last_pre_trace.zero_()
		self.last_post_trace.zero_()
		self.last_mem.zero_()
		self.last_surrogate.zero_()
		self.pre_decay_trace.zero_()
		self.post_decay_trace.zero_()
		self.mem_decay_trace.zero_()

	@torch.no_grad()
	def zero_elig(self):
		for name in self.params.keys():
			elig = getattr(self, f"{name}_elig")
			elig.zero_()

	@torch.no_grad()
	def forward(self, in_spikes: torch.Tensor) -> torch.Tensor:
		# decode actual parameter values
		pre_decay = torch.sigmoid(self.params["pre_decay"])  # [in]
		mem_decay = torch.sigmoid(self.params["mem_decay"])  # [out]
		threshold = torch.nn.functional.softplus(self.params["threshold"])  # [out]
		post_decay = torch.sigmoid(self.params["post_decay"])  # [out]

		# snapshot previous states for elig recurrences (use these in the recurrences)
		self.last_pre_trace.zero_().add_(self.pre_trace)
		self.last_post_trace.zero_().add_(self.post_trace)
		self.last_mem.zero_().add_(self.mem)

		# update pre-trace
		self.pre_trace.mul_(pre_decay).add_(in_spikes)

		# compute synaptic current using the param dict weight
		# weight is [out, in], in_spikes is [in] -> result [out]
		synaptic_current = torch.einsum("oi,i->o", self.params["weight"], in_spikes)

		# update membrane
		self.mem.mul_(mem_decay).add_(synaptic_current).add_(self.params["bias"])

		# spikes and instantaneous surrogate (before reset)
		out_spikes = (self.mem >= threshold).float()
		self.last_surrogate.zero_().add_(sigmoid_derivative(self.mem - threshold))

		# update post-trace
		self.post_trace.mul_(post_decay).add_(self.last_surrogate)

		# reset membrane where spikes happened
		self.mem.sub_(threshold * out_spikes)

		# ---- update the recurrence traces used for decay eligibilities ----
		# these hold things like ∂mem/∂mem_decay and ∂pre_trace/∂pre_decay
		self.pre_decay_trace.mul_(pre_decay).add_(self.last_pre_trace)  # [in]
		self.post_decay_trace.mul_(post_decay).add_(self.last_post_trace)  # [out]
		self.mem_decay_trace.mul_(mem_decay).add_(self.last_mem)  # [out]

		return out_spikes

	@torch.no_grad()
	def backward(self, learning_signal: torch.Tensor) -> torch.Tensor:
		# learning_signal shape: [out]

		# decode actual decays for normalization
		pre_decay = torch.sigmoid(self.params["pre_decay"])  # [in]
		post_decay = torch.sigmoid(self.params["post_decay"])  # [out]

		# normalized pre/post traces (average-style)
		normalized_pre = self.pre_trace * (1.0 - pre_decay)  # [in]
		normalized_post = self.post_trace * (1.0 - post_decay)  # [out]

		# ---------- weight: accumulate *real* gradient into weight_elig ----------
		# per-timestep outer(post, pre) => shape [out, in]
		weight_e_inc = torch.einsum("o,i->oi", normalized_post, normalized_pre)
		# multiply each output row by that output's learning signal and accumulate
		# shape preserved [out, in]
		self.weight_elig.add_(torch.einsum("oi,o->oi", weight_e_inc, learning_signal))

		# ---------- threshold: instantaneous effect is -surrogate ----------
		# threshold_elig stores gradient w.r.t. actual threshold (softplus output)
		self.threshold_elig.add_(learning_signal * (-self.last_surrogate))  # [out]

		# ---------- bias: direct current to membrane, effect on spike ~= last_surrogate ----------
		# bias_elig stores real gradient contributions for bias (actual-space)
		self.bias_elig.add_(learning_signal * self.last_surrogate)  # shape [out]

		# ---------- mem_decay: use the recurrence trace mem_decay_trace ----------
		# mem_decay_elig accumulates learning_signal * surrogate * mem_decay_trace
		self.mem_decay_elig.add_((learning_signal * self.last_surrogate) * self.mem_decay_trace)  # [out]

		# ---------- pre_decay: scalar S_post multiplies per-input pre_decay_trace ----------
		# S_post = sum_j learning_signal_j * normalized_post_j  (scalar)
		S_post = torch.dot(learning_signal, normalized_post)
		# each input's pre_decay gradient contribution is pre_decay_trace[i] * S_post
		self.pre_decay_elig.add_(self.pre_decay_trace * S_post)  # [in]

		# ---------- post_decay: for each output j: post_decay_trace[j] * learning_signal[j] * sum_pre ----------
		sum_pre = normalized_pre.sum()  # scalar
		self.post_decay_elig.add_((learning_signal * self.post_decay_trace) * sum_pre)  # [out]

		# ---------- compute upstream learning signal to pass to previous layer ----------
		gated = learning_signal * self.last_surrogate  # [out]
		passed_ls = torch.einsum("oi,o->i", self.params["weight"], gated)  # [in]

		return passed_ls

	@torch.no_grad()
	def elig_to_grad(self, scalar: float = 1.0):
		"""
		Move accumulated eligibilities (which are gradients w.r.t. actual parameter values)
		into the raw parameter .grad fields, applying transform derivatives.
		scalar: multiply elig by this scalar before converting (useful if you used fake adv=1).
		"""

		# helper to accumulate safely into p.grad
		def accum_grad(param, g_raw):
			if param.grad is None:
				param.grad = g_raw.clone()
			else:
				param.grad.add_(g_raw)

		# --- weight (identity transform) ---
		w = self.params["weight"]
		if self.weight_elig is not None:
			accum_grad(w, scalar * self.weight_elig)
			self.weight_elig.zero_()

		# --- bias (identity) ---
		raw = self.params["bias"]
		if getattr(self, "bias_elig", None) is not None:
			g_raw = scalar * self.bias_elig
			accum_grad(raw, g_raw)
			self.bias_elig.zero_()

		# --- mem_decay (actual = sigmoid(raw)): raw_grad = actual_grad * s * (1-s) ---
		raw = self.params["mem_decay"]
		s = torch.sigmoid(raw)
		g_raw = scalar * (self.mem_decay_elig * (s * (1.0 - s)))
		accum_grad(raw, g_raw)
		self.mem_decay_elig.zero_()

		# --- threshold (actual = softplus(raw)): raw_grad = actual_grad * sigmoid(raw) ---
		raw = self.params["threshold"]
		sigma_raw = torch.sigmoid(raw)  # derivative of softplus(raw) is sigmoid(raw)
		g_raw = scalar * (self.threshold_elig * sigma_raw)
		accum_grad(raw, g_raw)
		self.threshold_elig.zero_()

		# --- pre_decay (sigmoid) ---
		raw = self.params["pre_decay"]
		s = torch.sigmoid(raw)
		g_raw = scalar * (self.pre_decay_elig * (s * (1.0 - s)))
		accum_grad(raw, g_raw)
		self.pre_decay_elig.zero_()

		# --- post_decay (sigmoid) ---
		raw = self.params["post_decay"]
		s = torch.sigmoid(raw)
		g_raw = scalar * (self.post_decay_elig * (s * (1.0 - s)))
		accum_grad(raw, g_raw)
		self.post_decay_elig.zero_()
