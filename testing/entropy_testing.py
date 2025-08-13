import torch

import tracetorch as tt

n = 10
d = 0.6

trace = torch.zeros(n)
decay = tt.functional.sigmoid_inverse(torch.ones(n) * d)

entropy_manager = tt.plot.MeasurementManager(title="entropy")

num_timesteps = 1000

random_raw = torch.rand(n)

for _ in range(num_timesteps):
	random_bernoulli = torch.bernoulli(random_raw)
	trace = trace * decay + random_bernoulli
	avg_input = trace * (1 - decay)
	entropy, _ls = tt.loss.mse(avg_input, random_bernoulli)
	entropy_manager.append(entropy)

entropy_manager.plot()
