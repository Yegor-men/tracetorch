import torch

from .. import functional


class LIF:
	def __init__(
			self,
			n_in: int,
			n_out: int,
			mem_decay: float = 0.9,
			threshold: float = 1,
			config=None,
	):
		if config is None:
			config = {
				"device": "cuda",
				"lr": 1e-2,
			}
		self.device = config["device"]
		self.lr = config["lr"]

		self.weight = torch.randn(n_out, n_in).to(self.device)
		self.mem_decay = (functional.sigmoid_inverse(torch.ones(n_out) * mem_decay)).to(self.device)
		self.threshold = (functional.softplus_inverse(torch.ones(n_out) * threshold)).to(self.device)

		self.weight.requires_grad_(True)
		self.mem_decay.requires_grad_(True)
		self.threshold.requires_grad_(True)

		self.optimizer = torch.optim.AdamW([self.weight, self.mem_decay, self.threshold], lr=self.lr)

		self.mem = torch.zeros(n_out).to(self.device)
		self.in_trace = torch.zeros(n_in).to(self.device)
		self.in_trace_decay = 0.9

	def forward(self, in_spikes: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			in_spikes = in_spikes.to(self.device)
			self.in_trace = self.in_trace * self.in_trace_decay + in_spikes
			syn_current = torch.einsum("i, oi -> o", in_spikes, self.weight)
			mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
			self.mem = self.mem * mem_decay + syn_current
			threshold = torch.nn.functional.softplus(self.threshold)
			out_spikes = (self.mem >= threshold).float()
			self.mem -= threshold * out_spikes
			return out_spikes

	def backward(self, learning_signal: torch.Tensor) -> torch.Tensor:
		average_input = (self.in_trace * (1 - self.in_trace_decay)).detach().requires_grad_(True)
		average_syn_current = torch.einsum("i, oi -> o", average_input, self.weight)
		# if i=t(1-d), then t=i/(1-d)
		mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
		stabilized_mem = average_syn_current / (1 - mem_decay)
		threshold = torch.nn.functional.softplus(self.threshold)
		average_output = torch.nn.functional.sigmoid(stabilized_mem - threshold)

		average_output.backward(learning_signal)
		passed_ls = average_input.grad
		self.optimizer.step()
		self.optimizer.zero_grad(set_to_none=True)
		return passed_ls
