import torch
from torch import nn
from ... import functional
import math


class LIF(nn.Module):
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float = 0.9,
			threshold: float = 1,
			in_trace_decay: float = 0.9,
			learn_weight: bool = True,
			learn_mem_decay: bool = True,
			learn_threshold: bool = True,
			learn_in_trace_decay: bool = True,
	):
		super().__init__()
		self.num_in = int(num_in)
		self.num_out = int(num_out)

		self.learn_weight = bool(learn_weight)
		self.learn_mem_decay = bool(learn_mem_decay)
		self.learn_in_trace_decay = bool(learn_in_trace_decay)
		self.learn_threshold = bool(learn_threshold)

		t = threshold
		i = num_in
		self.weight = nn.Parameter(torch.normal(mean=t / i, std=t / math.sqrt(i), size=(num_out, num_in)))
		self.mem_decay = nn.Parameter(functional.sigmoid_inverse(torch.full((num_out,), mem_decay)))
		self.threshold = nn.Parameter(functional.softplus_inverse(torch.full((num_out,), threshold)))
		self.in_trace_decay = nn.Parameter(torch.full((num_in,), in_trace_decay))

		self.register_buffer("mem", torch.zeros(num_out))
		self.register_buffer("in_trace", torch.zeros(num_in))

	def get_learnable_parameters(self):
		learnable_parameters = [
			tensor
			for tensor, learnable in [
				(self.weight, self.learn_weight),
				(self.mem_decay, self.learn_mem_decay),
				(self.threshold, self.learn_threshold),
				(self.in_trace_decay, self.learn_in_trace_decay)
			] if learnable
		]

		return learnable_parameters

	@torch.no_grad()
	def zero_states(self):
		self.mem.zero_()
		self.in_trace.zero_()

	@torch.no_grad()
	def forward(self, in_spikes):
		in_trace_decay = nn.functional.sigmoid(self.in_trace_decay)
		self.in_trace.mul_(in_trace_decay).add_(in_spikes)
		synaptic_current = torch.einsum("i, oi -> o", in_spikes, self.weight)
		mem_decay = nn.functional.sigmoid(self.mem_decay)
		self.mem.mul_(mem_decay).add_(synaptic_current)
		threshold = nn.functional.softplus(self.threshold)
		out_spikes = (self.mem >= threshold).float()
		self.mem.addcmul_(out_spikes, threshold, value=-1.)
		return out_spikes

	def smooth_lif_rate_vector(self, i, d, t,
							   beta: float = 10.0,
							   k1: float = 30.0,
							   k2: float = 60.0,
							   eps: float = 1e-8):
		"""
		Vectorized smooth LIF firing frequency approximation for tensors i, d, t (same shape).
		All operations are differentiable and autograd-friendly.

		Args:
		  i: tensor of average inputs (shape [n])
		  d: tensor of decay factors in (0,1) (shape [n])
		  t: tensor of thresholds (>0 ideally) (shape [n])
		  beta: softplus stiffness (higher => closer to ReLU(i))
		  k1: "immediate-fire" gate sharpness (higher => sharper switch to f=1 when i >= t)
		  k2: "unreachable" gate sharpness (higher => sharper suppression when s -> 1)
		  eps: small numeric epsilon for stability

		Returns:
		  freq: tensor of shape [n] with values in [0,1] (smoothly differentiable)
		"""

		# ensure same device/dtype and numerical stability
		i = torch.as_tensor(i)
		d = torch.as_tensor(d)
		t = torch.as_tensor(t)

		# clamp d to (eps, 1-eps) to avoid log(0)
		d_clamped = d.clamp(min=eps, max=1.0 - eps)

		# soft positive input: i_pos = softplus(i, beta)
		# using torch.nn.functional.softplus with beta gives (1/beta) * ln(1 + exp(beta * x))
		i_pos = torch.nn.functional.softplus(i, beta=beta)

		# lambda = -ln(d)
		lam = -torch.log(d_clamped)

		# s = t * (1 - d) / (i_pos + eps)
		s = (t * (1.0 - d_clamped)) / (i_pos + eps)

		# safe "1 - s" with a smooth lower bound using softplus:
		# safe_one_minus_s = eps + softplus( (1 - s) - eps )
		one_minus_s = 1.0 - s
		safe_one_minus_s = eps + torch.nn.functional.softplus(one_minus_s - eps)

		# denom = -ln(safe_one_minus_s)  (guaranteed > 0)
		denom = -torch.log(safe_one_minus_s + eps)

		# raw analytic frequency (may be > 1 or small)
		raw = lam / (denom + eps)

		# immediate-fire gate: if i_pos >> t -> g_i ~ 1 -> freq ≈ 1
		g_i = torch.sigmoid(k1 * (i_pos - t))

		# reachability gate: suppress when s approaches 1 (we want g_s ~ 0 near s >= 1)
		# choose pivot close to 1 (e.g., 0.999) so we smoothly go to zero for s near 1
		pivot = 0.999
		g_s = torch.sigmoid(k2 * (pivot - s))  # ~1 when s << pivot, ~0 when s >> pivot

		# combine:
		# - if i_pos >> t -> g_i ~ 1 so freq ~ 1
		# - else use (g_s * raw) to smoothly go to raw in valid region, and to 0 when s >= 1
		freq = g_i * 1.0 + (1.0 - g_i) * (g_s * raw)

		# final safety: clamp into [0,1]
		freq = freq.clamp(min=0.0, max=1.0)

		return freq

	def backward(self, learning_signal):
		in_trace_decay = nn.functional.sigmoid(self.in_trace_decay)
		average_input = self.in_trace * (1 - in_trace_decay)
		average_input.retain_grad()
		i = torch.einsum("i, oi -> o", average_input, self.weight)
		d = nn.functional.sigmoid(self.mem_decay)
		t = nn.functional.softplus(self.threshold)

		# excess = (2 * i - i * d) / 2 - t * (1 - d)
		# frequency = torch.nn.functional.sigmoid(excess)

		frequency = self.smooth_lif_rate_vector(i, d, t)

		frequency.backward(learning_signal.detach())
		passed_learning_signal = average_input.grad.detach().clone()
		average_input.grad = None
		del average_input
		return passed_learning_signal