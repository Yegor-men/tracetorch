import torch

from .. import functional


class LIF:
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
		self.threshold = (functional.softplus_inverse(torch.ones(n_out) * threshold)).to(self.device)

		self.weight.requires_grad_(True)
		self.mem_decay.requires_grad_(True)
		self.in_trace_decay.requires_grad_(True)
		self.threshold.requires_grad_(True)

		self.optimizer = torch.optim.AdamW(
			[self.weight, self.mem_decay, self.in_trace_decay, self.threshold],
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
			threshold = torch.nn.functional.softplus(self.threshold)
			out_spikes = (self.mem >= threshold).float()
			self.mem -= threshold * out_spikes
			return out_spikes

	def backward(self, learning_signal: torch.Tensor) -> torch.Tensor:
		in_trace_decay = torch.nn.functional.sigmoid(self.in_trace_decay)
		average_input = self.in_trace * (1 - in_trace_decay)
		avg_in = average_input.detach().clone().requires_grad_(True)

		i = torch.einsum("i, oi -> o", avg_in, self.weight)
		d = torch.nn.functional.sigmoid(self.mem_decay)
		t = torch.nn.functional.softplus(self.threshold)

		will_fire = t < i / (1 - d)

		valid_freq = torch.log(d) / torch.log(1 - (t / i) * (1 - d))
		invalid_freq = 1 / (torch.exp(i + i * d) * torch.exp(t - i / (1 - d)))

		f = torch.where(will_fire, valid_freq, invalid_freq)

		f.backward(learning_signal)
		passed_ls = avg_in.grad
		average_input.backward(passed_ls)

		self.optimizer.step()
		self.optimizer.step()
		self.optimizer.zero_grad(set_to_none=True)
		print(f"wil fire: {will_fire}")
		print(f"avg outp: {f}")
		print(f"passedls: {passed_ls}")
		return passed_ls
