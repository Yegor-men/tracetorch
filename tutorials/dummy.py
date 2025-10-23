import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

in_features, hidden_features, out_features = 100, 128, 3

model = snn.Sequential(
	nn.Linear(in_features, hidden_features),
	snn.Synaptic(hidden_features, 0.2, 0.9, 1.0),
	nn.Linear(hidden_features, hidden_features),
	snn.Synaptic(hidden_features, 0.2, 0.9, 1.0),
	nn.Linear(hidden_features, out_features),
	snn.Readout(out_features, 0.9, learn_beta=True, beta_is_scalar=True),
	nn.Softmax(-1),
).to(device)

rand_in = torch.rand(in_features).to(device)
rand_out = nn.functional.softmax(torch.randn(out_features), -1).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_timesteps = 100
num_epochs = 100

loss_manager = tt.plot.MeasurementManager("Train Loss", [0.0, 0.5, 0.9, 0.99])

for epoch in tqdm(range(num_epochs), total=num_epochs):
	model.zero_states()
	model.zero_grad()

	for t in range(num_timesteps):
		spike_input = torch.bernoulli(rand_in)
		model_out = model(spike_input)
		loss = nn.functional.mse_loss(model_out, rand_out)

	loss_manager.append(loss.item())
	loss.backward()
	optimizer.step()

loss_manager.plot()

spike_train = []
loss_train = []

with torch.no_grad():
	for t in tqdm(range(num_timesteps), total=num_timesteps):
		spike_input = torch.bernoulli(rand_in)
		model_out = model(spike_input)
		spike_train.append(model_out)
		loss = nn.functional.mse_loss(model_out, rand_out)
		loss_train.append(loss.item())

tt.plot.spike_train(spike_train, title="Spikes over time")
tt.plot.line_graph(spike_train, title="Output over time")
tt.plot.line_graph(loss_train, title="Loss over time")

print(f"Exp: {rand_out}")
print(f"Got: {model_out}")
print(f"Final timestep loss: {loss_train[-1]}")

print(nn.functional.sigmoid(model.layers[-2].beta))