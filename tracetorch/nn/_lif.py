import torch

from .. import functional


class LIF:
	"""
	Leaky integrate and fire neuron layer
	"""

	def __init__(
			self,
			num_in: int,
			num_out: int,
			weight_scaling: float = 0.1,
			mem_decay: float = 0.9,
			in_trace_decay: float = 0.9,
			threshold: float = 1,
			device: str = "cpu",
			lr: float = 1e-3,
			learn_weight: bool = True,
			learn_mem_decay: bool = True,
			learn_in_trace_decay: bool = True,
			learn_threshold: bool = True
	):
		self.device = device
		self.lr = lr

		self.weight = (torch.randn(num_out, num_in) * weight_scaling).to(self.device)
		self.mem_decay = (functional.sigmoid_inverse(torch.ones(num_out) * mem_decay)).to(self.device)
		self.in_trace_decay = (functional.sigmoid_inverse(torch.ones(num_in) * in_trace_decay)).to(self.device)
		self.threshold = (functional.softplus_inverse(torch.ones(num_out) * threshold)).to(self.device)
		self.mem = torch.zeros(num_out).to(self.device)
		self.in_trace = torch.zeros(num_in).to(self.device)

		self.all_tensors = [
			self.weight,
			self.mem_decay,
			self.in_trace_decay,
			self.threshold,
			self.mem,
			self.in_trace
		]

		for tensor in [self.weight, self.mem_decay, self.in_trace_decay, self.threshold]:
			tensor.requires_grad_(True)

		self.learnable_parameters = [
			t for t, learn in [
				(self.weight, learn_weight),
				(self.mem_decay, learn_mem_decay),
				(self.in_trace_decay, learn_in_trace_decay),
				(self.threshold, learn_threshold)
			] if learn
		]

		self.optimizer = torch.optim.AdamW(self.learnable_parameters, lr=self.lr)

	def clear_grads(self):
		for t in self.all_tensors:
			if t.grad is not None:
				t.grad = None

	def forward(self, in_spikes: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			in_spikes = in_spikes.to(self.device)
			in_trace_decay = torch.nn.functional.sigmoid(self.in_trace_decay)
			self.in_trace = self.in_trace * in_trace_decay + in_spikes
			syn_current = torch.einsum("i, oi -> o", in_spikes, self.weight)
			mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
			self.mem = self.mem * mem_decay + syn_current
			threshold = torch.nn.functional.softplus(self.threshold)
			out_spikes = (self.mem >= threshold).float()
			self.mem -= threshold * out_spikes
			return out_spikes

	def backward(self, learning_signal: torch.Tensor) -> torch.Tensor:
		in_trace_decay = torch.nn.functional.sigmoid(self.in_trace_decay)
		average_input = self.in_trace * (1 - in_trace_decay)
		avg_in = average_input.detach().requires_grad_(True)

		i = torch.einsum("i, oi -> o", avg_in, self.weight)
		d = torch.nn.functional.sigmoid(self.mem_decay)
		t = torch.nn.functional.softplus(self.threshold)

		excess = (2 * i - i * d) / 2 - t * (1 - d)
		f = torch.nn.functional.sigmoid(5 * excess)

		f.backward(learning_signal)
		passed_ls = avg_in.grad
		average_input.backward(passed_ls)

		self.optimizer.step()
		self.clear_grads()
		return passed_ls

	def zero_states(self):
		with torch.no_grad():
			self.in_trace.zero_()
			self.mem.zero_()
