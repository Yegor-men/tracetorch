import torch
from torch import nn
from .. import functional


class Reflect(nn.Module):
	def __init__(
			self,
			num_in: int,
			decay: float = 0.9,
			learn_decay: bool = True
	):
		super().__init__()
		self.num_in = num_in

		self.learn_decay = learn_decay

		self.decay = nn.Parameter(functional.sigmoid_inverse(torch.tensor(decay)))

		self.register_buffer("logprob_trace", torch.zeros(num_in))
		self.register_buffer("output_trace", torch.zeros(num_in))
		self.register_buffer("reward_trace", torch.tensor(0.))

	def get_learnable_parameters(self):
		learnable_parameters = [
			t for t, learn in [
				(self.decay, self.learn_decay),
			] if learn
		]
		return learnable_parameters

	def zero_states(self, clear_logprob=True, clear_output=True, clear_reward=True):
		for trace, clear in [
			(self.logprob_trace, clear_logprob),
			(self.output_trace, clear_output),
			(self.reward_trace, clear_reward)
		]:
			if clear:
				trace.zero_()

	def forward(self, distribution, output):
		with torch.no_grad():
			d = torch.nn.functional.sigmoid(self.decay)
			self.logprob_trace.mul_(d).add_(torch.log(distribution + 1e-12))
			self.output_trace.mul_(d).add_(output)

	def backward(self, reward):
		with torch.no_grad():
			baseline_reward = self.reward_trace * (1 - torch.nn.functional.sigmoid(self.decay))
			self.reward_trace.mul_(torch.nn.functional.sigmoid(self.decay)).add_(reward)

		d = torch.nn.functional.sigmoid(self.decay)

		advantage = reward - baseline_reward

		average_logprob = self.logprob_trace * (1 - d)
		average_logprob.retain_grad()
		average_output = self.output_trace * (1 - d)

		loss = -advantage * (average_output * average_logprob).sum()
		# learning_signal = (-advantage * average_logprob).detach().clone()

		loss.backward()
		passed_ls = average_logprob.grad.detach().clone()
		average_logprob.grad = None
		del average_logprob
		return loss, passed_ls
