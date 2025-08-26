import torch


def actor_critic(
		distribution: torch.Tensor,
		sample: torch.Tensor,
		value,
		next_value,
		reward,
		gamma=0.99
):
	eps = 1e-9

	distribution = distribution.detach().clone().clamp(min=eps).requires_grad_(True)

	logprob = (torch.log(distribution) * sample).sum()

	td_error = reward + gamma * next_value - value

	actor_loss = -td_error.detach().clone() * logprob
	actor_loss.backward()
	actor_ls = distribution.grad.detach().clone()

	critic_target = reward + gamma * next_value.detach().clone()
	critic_loss = torch.nn.functional.mse_loss(value, critic_target)

	return actor_loss, actor_ls, critic_loss
