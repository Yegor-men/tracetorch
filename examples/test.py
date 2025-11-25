import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class SNN(snn.TTModule):
	def __init__(self, c, n_labels):
		super().__init__()

		self.mlp = nn.Sequential(
			nn.Conv2d(c, 16, 3),
			snn.LeakyIntegrator(16, beta_setup={}, pos_threshold_setup={}, dim=-3),
			nn.Flatten(),
			nn.LazyLinear(16),
			snn.LeakyIntegrator(16, beta_setup={}, pos_threshold_setup={}, neg_threshold_setup={}, gamma_setup={},
								weight_setup={}),
			nn.Linear(16, 16),
			snn.LeakyIntegrator(16, beta_setup={}, pos_threshold_setup={}),
			nn.Linear(16, 16),
			snn.LeakyIntegrator(16, beta_setup={}, pos_threshold_setup={}),
			nn.Linear(16, n_labels),
			snn.LeakyIntegrator(n_labels, beta_setup={"rank": 0, "use_averaging": True}),
			nn.Softmax(-1)
		)

	def forward(self, x):
		return self.mlp(x)


b, c, h, w = 10, 3, 28, 28

model = SNN(c, b).to(device)

random_feature = torch.rand(b, c, h, w).to(device)
random_label = torch.eye(b).to(device)

print(f"Feature: {random_feature.shape} | Label: {random_label.shape}")

num_epochs = 100
num_timesteps = 20

optimizer = torch.optim.AdamW(model.parameters(), 1e-2)

losses = []

# loss_fn = nn.functional.mse_loss
loss_fn = tt.loss.soft_cross_entropy  # the more flexible variant of cross-entropy for not only onehot vectors

for epoch in tqdm(range(num_epochs), desc="TRAIN", total=num_epochs):
	model.zero_grad()
	model.train()
	for step in range(num_timesteps):
		spiked_input = torch.bernoulli(random_feature)
		model_output = model(spiked_input)
	loss = loss_fn(model_output, random_label)
	loss.backward()
	losses.append(loss.item())
	optimizer.step()
	model.zero_states()  # despite the name, actually sets them to None

plt.title("Train loss over time")
plt.plot(losses, label="Train Loss")
plt.legend()
plt.show()

# Let's also take a look at the spike train output that the model makes for each example:
spike_train = []

with torch.no_grad():
	model.eval()
	model.zero_states()
	for step in tqdm(range(num_timesteps), desc="TEST", total=num_timesteps):
		spiked_input = torch.bernoulli(random_feature)
		model_output = model(spiked_input)
		spike_train.append(model_output)

	for index in range(b):
		curated_spike_train = [tensor[index] for tensor in spike_train]
		curated_loss_train = [loss_fn(tensor[index], random_label[index]) for tensor in spike_train]

		print(f"Example {index} - Expected output: {random_label[index]}")
		tt.plot.line_graph(curated_spike_train, title=f"Example {index} - Output over time")
# tt.plot.line_graph(curated_loss_train, title=f"Example {index} - Loss over time")
