import matplotlib.pyplot
import torch
import tracetorch

config_dict = {
	"device": "cuda",
	"lr": 1e-3,
}

n_in = 10
n_out = 40
n_hidden = 10

model = tracetorch.nn.Sequential(
	tracetorch.nn.LIF(
		n_in=n_in,
		n_out=n_hidden,
		config=config_dict
	),
	tracetorch.nn.LIF(
		n_in=n_hidden,
		n_out=n_hidden,
		config=config_dict
	),
	tracetorch.nn.LIF(
		n_in=n_hidden,
		n_out=n_hidden,
		config=config_dict
	),
	tracetorch.nn.LIF(
		n_in=n_hidden,
		n_out=n_out,
		config=config_dict
	),
)

n_samples = 5

samples = [(torch.rand(n_in).round(), torch.rand(n_out).round()) for _ in range(n_samples)]

n_epochs = 100
think_length = 10

losses = []

for epoch in range(n_epochs):
	for index, (x, y) in enumerate(samples):
		model.zero_states()
		cum_loss = 0
		for j in range(think_length):
			model_output = model.forward(x)
			loss, ls = tracetorch.loss.mse(model_output, y)
			cum_loss += loss
			if j == think_length - 1:
				model.backward(ls)
		cum_loss /= think_length
		losses.append(cum_loss)
		print(f"Epoch: {epoch:,}, Sample: {index} - Loss: {cum_loss}")

tracetorch.plot.line_graph(losses, "loss")

for index, (x, y) in enumerate(samples):
	model.zero_states()
	model_output_list = []
	cum_loss = 0
	for j in range(think_length):
		model_output = model.forward(x)
		model_output_list.append(model_output)
		loss, ls = tracetorch.loss.mse(model_output, y)
		cum_loss += loss
	cum_loss /= think_length
	tracetorch.plot.spike_train(model_output_list, title=f"Index: {index}, Loss: {cum_loss}")
