import torch

from .. import functional


class Softmax:
	def __init__(
			self,
			n_in: int,
			n_out: int,
			mem_decay: float = 0.9,
			in_trace_decay: float = 0.9,
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
		self.in_trace_decay = (functional.sigmoid_inverse(torch.ones(n_in) * in_trace_decay)).to(self.device)

		self.weight.requires_grad_(True)
		self.mem_decay.requires_grad_(True)
		self.in_trace_decay.requires_grad_(True)

		self.optimizer = torch.optim.AdamW(
			[self.weight, self.mem_decay, self.in_trace_decay],
			lr=self.lr
		)

		self.mem = torch.zeros(n_out).to(self.device)
		self.in_trace = torch.zeros(n_in).to(self.device)

	def forward(self, in_spikes: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			in_spikes = in_spikes.to(self.device)
			in_trace_decay = torch.nn.functional.sigmoid(self.in_trace_decay)
			self.in_trace = self.in_trace * in_trace_decay + in_spikes
			syn_current = torch.einsum("i, oi -> o", in_spikes, self.weight)
			mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
			self.mem = self.mem * mem_decay + syn_current
			probability_distribution = torch.nn.functional.softmax(self.mem, dim=-1)
			index = torch.multinomial(probability_distribution, num_samples=1)
			out_spikes = torch.zeros_like(probability_distribution)
			out_spikes[index] = 1.
			return out_spikes

	def backward(self, learning_signal: torch.Tensor) -> torch.Tensor:
		with torch.no_grad():
			in_trace_decay = torch.nn.functional.sigmoid(self.in_trace_decay)
		average_input = (self.in_trace * (1 - in_trace_decay)).detach().requires_grad_(True)
		average_syn_current = torch.einsum("i, oi -> o", average_input, self.weight)
		# if i=t(1-d), then t=i/(1-d)
		mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
		stabilized_mem = average_syn_current / (1 - mem_decay)
		average_output = torch.nn.functional.softmax(stabilized_mem, dim=-1)

		average_output.backward(learning_signal)
		passed_ls = average_input.grad

		in_trace_decay = torch.nn.functional.sigmoid(self.in_trace_decay)
		avg_in = (self.in_trace * (1 - in_trace_decay))
		avg_in.backward(passed_ls)

		self.optimizer.step()
		self.optimizer.zero_grad(set_to_none=True)
		return passed_ls
