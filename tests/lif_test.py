import matplotlib.pyplot
import torch
import tracetorch

config_dict = {
	"device": "cuda",
	"lr": 1e-2,
}

n_in = 10000
n_out = 10000
n_hidden = 10000

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

random_x = torch.rand(n_in).round()
random_y = torch.rand(n_out).round()

n_timesteps = 50
model_outputs = []
losses = []
lses = []

for i in range(n_timesteps):
	model_out = model.forward(random_x)
	model_outputs.append(model_out)

	loss, ls = tracetorch.loss.mse(model_out, random_y)
	print(f"{i} - Loss: {loss}")

	losses.append(loss)
	lses.append(ls)

	model.backward(ls)

tracetorch.plot.spike_train(model_outputs)
tracetorch.plot.line_graph(losses, "loss")
tracetorch.plot.line_graph(lses, "ls")

# print(f"Got: {model_outputs[-1]}")
# print(f"Exp: {random_y}")

# with torch.no_grad():
# 	print(f"{torch.nn.functional.sigmoid(model.layers[0].mem_decay)}")
# 	print(f"{torch.nn.functional.softplus(model.layers[0].threshold)}")
# 	print(f"{torch.nn.functional.sigmoid(model.layers[0].in_trace_decay)}")
