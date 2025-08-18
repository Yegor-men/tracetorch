import torch
from torch import nn
from .. import functional


class ReflectMTF(nn.Module):
	def __init__(
			self,
			num_in: int,
			decay: list = [0.5, 0.9, 0.99, 0.999],
			learn_weight: bool = True,
	):
		super().__init__()
		self.num_in = int(num_in)
		self.learn_weight = bool(learn_weight)

		self.register_buffer("decay", torch.tensor(decay))
		self.weight = nn.Parameter(torch.zeros_like(self.decay))
		self.k = self.weight.numel()
		self.register_buffer("logprob_trace", torch.zeros(self.k, num_in))
		self.register_buffer("output_trace", torch.zeros(self.k, num_in))
		self.register_buffer("reward_trace", torch.zeros(self.k))

	def get_learnable_parameters(self):
		learnable_parameters = [
			t for t, learn in [
				(self.weight, self.learn_weight),
			] if learn
		]
		return learnable_parameters

	def zero_states(self, clear_traces=True, clear_reward=False):
		for trace, clear in [
			(self.logprob_trace, clear_traces),
			(self.output_trace, clear_traces),
			(self.reward_trace, clear_reward)
		]:
			if clear:
				trace.zero_()

	def forward(self, distribution, output):
		with torch.no_grad():
			for k, decay in enumerate(self.decay):
				self.logprob_trace[k].mul_(decay).add_(torch.log(distribution + 1e-12))
				self.output_trace[k].mul_(decay).add_(output)

	def backward(self, reward):
		with torch.no_grad():
			baseline_rewards = self.reward_trace * (1 - self.decay)
			self.reward_trace.mul_(self.decay).add_(reward)
			smax_scale = torch.softmax(self.weight, dim=-1)
			baseline_reward = torch.einsum("k, k ->", baseline_rewards, smax_scale)

		softmax_scaling = torch.softmax(self.weight, dim=-1)

		advantage = reward - baseline_reward

		avg_logprobs = torch.einsum("ko, k -> ko", self.logprob_trace, (1 - self.decay))
		avg_outputs = torch.einsum("ko, k -> ko", self.output_trace, (1 - self.decay))

		average_logprob = torch.einsum("ko, k -> o", avg_logprobs, softmax_scaling)
		average_output = torch.einsum("ko, k -> o", avg_outputs, softmax_scaling)
		average_logprob.retain_grad()

		loss = -advantage * (average_output * average_logprob).sum()

		loss.backward()
		passed_ls = average_logprob.grad.detach().clone()
		average_logprob.grad = None
		del average_logprob
		return loss, passed_ls
