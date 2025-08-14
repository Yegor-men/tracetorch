import torch
from tracetorch import functional


class Reflect:
	def __init__(
			self,
			num_out: int,
			decay: float = 0.9,
			device: str = "cpu",
			lr: float = 1e-3,
			learn_decay: bool = True
	):
		self.device = device
		self.lr = lr

		self.distribution_trace = torch.zeros(num_out).to(device)
		self.output_trace = torch.zeros(num_out).to(device)
		self.decay = functional.sigmoid_inverse(torch.tensor(decay)).to(device)
		self.reward_trace = torch.tensor(0.).to(device)

		self.all_tensors = [self.distribution_trace, self.output_trace, self.decay, self.reward_trace]

		for tensor in self.all_tensors:
			tensor.requires_grad_(True)

		self.learnable_parameters = [
			t for t, learn in [
				(self.decay, learn_decay),
			] if learn
		]

		self.optimizer = torch.optim.AdamW(self.learnable_parameters, lr=lr)

	def clear_grads(self):
		for t in self.all_tensors:
			if t.grad is not None:
				t.grad = None

	def forward(self, distribution, output):
		with torch.no_grad():
			self.distribution_trace = self.distribution_trace * torch.nn.functional.sigmoid(self.decay) + distribution
			self.output_trace = self.output_trace * torch.nn.functional.sigmoid(self.decay) + output

	def backward(self, reward):
		with torch.no_grad():
			baseline_reward = self.reward_trace * (1 - torch.nn.functional.sigmoid(self.decay))
			self.reward_trace = self.reward_trace * torch.nn.functional.sigmoid(self.decay) + reward

		d = torch.nn.functional.sigmoid(self.decay)

		advantage = reward - baseline_reward

		average_distribution = self.distribution_trace * (1 - d)
		average_distribution.retain_grad()
		average_output = self.output_trace * (1 - d)

		loss = -advantage * (average_output * torch.log(average_distribution + 1e-12)).sum()

		loss.backward()
		passed_ls = average_distribution.grad

		self.optimizer.step()
		self.clear_grads()
		return passed_ls
