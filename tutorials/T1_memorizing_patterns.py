import torch
import tracetorch as tt
import random

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_in = 10
n_hidden = 40
n_out = 10

data = []

for i in range(n_out):
	x = torch.rand(n_in)
	y = torch.zeros(n_out)
	y[i] = 1.
	data.append((x, y))

model = tt.nn.Sequential(
	tt.nn.LIF(
		num_in=n_in,
		num_out=n_hidden,
		device=device
	),
	tt.nn.LIF(
		num_in=n_hidden,
		num_out=n_hidden,
		device=device
	),
	tt.nn.LIS(
		num_in=n_hidden,
		num_out=n_out,
		device=device
	)
)

think_steps = 50
num_epochs = 100

train_loss_manager = tt.plot.MeasurementManager(title="Loss")
train_accuracy_manager = tt.plot.MeasurementManager(title="Accuracy")

for epoch in range(num_epochs):
	print(f"Epoch: {epoch + 1}/{num_epochs} - {((epoch + 1) / num_epochs) * 100:.2f}%")
	random.shuffle(data)
	for index, (x, y) in enumerate(data):
		for step in range(think_steps):
			model_out = model.forward(torch.bernoulli(x))
		loss, ls = tt.loss.mse(model_out, y)
		model.backward(ls)
		train_loss_manager.append(loss)
		train_accuracy_manager.append(1 if model_out.argmax().item() == y.argmax().item() else 0)

train_loss_manager.plot()
train_accuracy_manager.plot()

for index, (x, y) in enumerate(data):
	inputs = []
	distributions = []
	outputs = []
	for step in range(think_steps):
		model_input = torch.bernoulli(x)
		model_out = model.forward(model_input)
		model_choice = tt.functional.sample_softmax(model_out)
		inputs.append(model_input)
		distributions.append(model_out)
		outputs.append(model_choice)
	tt.plot.spike_train(inputs, title=f"{index}, Inputs over time")
	tt.plot.spike_train(distributions, title=f"{index}, Distribution over time")
	tt.plot.spike_train(outputs, title=f"{index}, Choices made")

in_trace_decays = [torch.nn.functional.sigmoid(layer.in_trace_decay) for layer in model.layers]
mem_decays = [torch.nn.functional.sigmoid(layer.mem_decay) for layer in model.layers]
weights = [layer.weight for layer in model.layers]

tt.plot.distributions(in_trace_decays, title="Input trace decay")
tt.plot.distributions(mem_decays, title="Membrane decay")
tt.plot.distributions(weights, title="Weights")
