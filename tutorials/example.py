import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn
from tqdm import tqdm


class Model(nn.Module):
	def __init__(
			self,
			in_features: int,
			middle_features: int,
			out_features: int,
	):
		super().__init__()

		self.foo = snn.Sequential(
			nn.Linear(in_features, middle_features),
			snn.Synaptic(middle_features, 0.5, 0.5, 1.0, view_tuple=(-1)),
			nn.Linear(middle_features, middle_features),
			snn.Synaptic(middle_features, 0.5, 0.5, 1.0, view_tuple=(-1)),
			nn.Linear(middle_features, out_features),
			snn.Synaptic(out_features, 0.5, 0.5, 1.0, return_mem=False, view_tuple=(-1)),
			# nn.Softmax(-1)
		)

	def zero_states(self):
		self.foo.zero_states()

	def detach_states(self):
		self.foo.detach_states()

	def forward(self, x):
		return self.foo(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

I, H, O = 100, 128, 50

model = Model(I, H, O).to(device)
rand_in = torch.rand(I).to(device)
# rand_out = nn.functional.softmax(torch.randn(O), -1).to(device)
rand_out = torch.rand(O).round().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

num_timesteps = 100
num_epochs = 100

losses = []

for e in tqdm(range(num_epochs), total=num_epochs):
	model.zero_states()
	model.zero_grad()

	for t in range(num_timesteps):
		spike_input = torch.bernoulli(rand_in)
		model_out = model(spike_input)
		loss = nn.functional.mse_loss(model_out, rand_out)
		losses.append(loss.item())
		loss.backward()
		optimizer.step()
		model.detach_states()

import matplotlib.pyplot as plt

plt.plot(losses, label="Loss")
plt.legend()
plt.title("Train")
plt.show()

example_losses = []

with torch.no_grad():
	average_out = torch.zeros(O).to(device)

	for t in tqdm(range(num_timesteps), total=num_timesteps):
		spike_input = torch.bernoulli(rand_in)
		model_out = model(spike_input)
		average_out += model_out
		loss = nn.functional.mse_loss(model_out, rand_out)
		example_losses.append(loss.item())

	average_out /= num_timesteps

print(f"Gotten: {average_out}")
print(f"Expect: {rand_out}")
print(f"Loss: {nn.functional.mse_loss(average_out, rand_out)}")

plt.plot(example_losses, label="Loss")
plt.legend()
plt.title("Inspect")
plt.show()
