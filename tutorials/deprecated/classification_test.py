import torch
import tracetorch
import random

config_dict = {
	"device": "cuda",
	"lr": 1e-3,
}

n_in = 10
n_hidden = 40
n_out = 10

model = tracetorch.nn.Sequential(
	tracetorch.nn.LIF(
		num_in=n_in,
		num_out=n_hidden,
		config=config_dict
	),
	tracetorch.nn.LIF(
		num_in=n_hidden,
		num_out=n_hidden,
		config=config_dict
	),
	tracetorch.nn.LIF(
		num_in=n_hidden,
		num_out=n_hidden,
		config=config_dict
	),
	tracetorch.nn.LIS(
		n_in=n_hidden,
		n_out=n_out,
		config=config_dict
	),
)

samples = []

for i in range(n_out):
	x = torch.rand(n_in).round()
	y = torch.zeros(n_out)
	y[i] = 1.
	samples.append((x, y))

n_epochs = 50
think_length = 50

losses = []

for epoch in range(n_epochs):
	random.shuffle(samples)
	for index, (x, y) in enumerate(samples):
		model.zero_states()
		for j in range(think_length):
			output_distribution = model.forward(x)
			model_output = tracetorch.functional.sample_softmax(output_distribution)

		loss, ls = tracetorch.loss.mse(output_distribution, y)
		model.backward(ls)
		losses.append(loss)
		print(f"Epoch: {epoch:,}, Sample: {index} - Loss: {loss}")

tracetorch.plot.line_graph(losses, "loss")

for index, (x, y) in enumerate(samples):
	model.zero_states()
	model_out_aggregate = torch.zeros_like(model.layers[-1].mem)
	model_output_list = []
	for j in range(think_length):
		model_output = model.forward(x)
		model_output_list.append(model_output)
		model_out_aggregate += model_output

	model_out_aggregate /= model_out_aggregate.sum()
	loss, ls = tracetorch.loss.mse(model_out_aggregate, y)
	tracetorch.plot.spike_train(model_output_list, title=f"Index: {index}, Loss: {loss}")
