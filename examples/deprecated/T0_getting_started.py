"""
A brute memorization of patterns, some input (spikes or raw floats) must map to some output softmaxed distribution

This is only used as a proof of concept; if a model cannot memorize, it cannot generalize

Training is tested for both BPTT and online. BPTT builds the computation graph across all timesteps, and online learning
detaches the states at each timestep so that it's discrete from one another
"""

# ======================================================================================================================

import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================================================================

in_features, hidden_features, out_features = 100, 128, 3

batch_size = 10

feature_tensor = torch.rand(batch_size, in_features).to(device=device)
label_tensor = torch.distributions.Dirichlet(torch.full((out_features,), 1.0)).sample((batch_size,)).to(device=device)

print(f"Input size: {feature_tensor.size()}")
print(f"Output size: {label_tensor.size()}")

beta = 0.9  # how much of the signal from the previous timestep is saved into this one

model = snn.Sequential(
	nn.Linear(in_features, hidden_features),
	snn.LIF(hidden_features, beta),
	nn.Linear(hidden_features, hidden_features),
	snn.LIF(hidden_features, beta),
	nn.Linear(hidden_features, out_features),
	snn.Readout(out_features, beta),
	nn.Softmax(-1)
).to(device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = 100  # how many optimizer steps will happen
num_timesteps = 100  # how many timesteps the input is flashed to the model

train_bptt = True
train_online = not train_bptt

loss_manager = tt.plot.MeasurementManager(f"Train Loss for {"BPTT" if train_bptt else "online"}", [0.0])

for e in tqdm(range(num_epochs), total=num_epochs, desc=f"Training, {"BPTT" if train_bptt else "online"}"):
	model.zero_states()
	model.zero_grad()

	if train_bptt:
		for t in range(num_timesteps):
			spike_input = torch.bernoulli(feature_tensor)
			model_output = model(spike_input)
			loss = tt.loss.soft_cross_entropy(model_output, label_tensor)

		loss_manager.append(loss.item())
		loss.backward()
		optimizer.step()

	if train_online:
		# for the sake of simplicity, it's assumed that we're only interested in the last timestep being correct
		for t in range(num_timesteps):
			spike_input = torch.bernoulli(feature_tensor)
			model_output = model(spike_input)
			loss = tt.loss.soft_cross_entropy(model_output, label_tensor)
			loss.backward()
			model.detach_states()

		loss_manager.append(loss.item())
		with torch.no_grad():
			for param in model.parameters():
				param.grad.div_(num_timesteps)
		optimizer.step()

loss_manager.plot()

with torch.no_grad():
	spike_train = []

	for t in tqdm(range(num_timesteps), total=num_timesteps, desc="Rendering"):
		spike_input = torch.bernoulli(feature_tensor)
		model_output = model(spike_input)
		spike_train.append(model_output)

	# loss_train = [tt.loss.soft_cross_entropy(tensor, label_tensor) for tensor in spike_train]

	for index in range(out_features):
		curated_spike_train = [tensor[index] for tensor in spike_train]
		curated_loss_train = [tt.loss.soft_cross_entropy(tensor[index], label_tensor[index]) for tensor in spike_train]

		print(f"Example {index} - Expected output: {label_tensor[index]}")
		tt.plot.line_graph(curated_spike_train, title=f"Example {index} - Output over time")
		tt.plot.line_graph(curated_loss_train, title=f"Example {index} - Loss over time")

betas = model.get_attr_list("beta")
for beta in betas:
	print(beta)

thresholds = model.get_attr_list("threshold")
for threshold in thresholds:
	print(threshold)
