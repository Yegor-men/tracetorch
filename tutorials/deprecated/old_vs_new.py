import torch
import tracetorch as tt
from tracetorch import neurons
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

min_prob = 0
max_prob = 1

# Flattens the image and redistributes the brightness domain from [0, 1] to [min_prob, max_prob]
image_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x.view(-1)),
	transforms.Lambda(lambda x: x * (max_prob - min_prob) + min_prob)
])


def one_hot_encode(y):
	return torch.nn.functional.one_hot(torch.tensor(y), num_classes=10).float()


# Custom collate function to remove batch dimension so each sample is returned as tensors [784] and [10]
def collate_fn(batch):
	image, label = batch[0]
	return image.squeeze(), label.squeeze()


dataset_kwargs = dict(root='data', download=True,
					  transform=image_transform,
					  target_transform=one_hot_encode)
train_dataset = datasets.MNIST(train=True, **dataset_kwargs)
test_dataset = datasets.MNIST(train=False, **dataset_kwargs)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ______________________________________________________________________________________________________________________
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

num_hidden = 32

old_model = tt.legacy.snn.Sequential(
	tt.legacy.snn.LIF(
		num_in=784,
		num_out=num_hidden,
	),
	tt.legacy.snn.LIF(
		num_in=num_hidden,
		num_out=num_hidden,
	),
	tt.legacy.snn.LIS(
		num_in=num_hidden,
		num_out=10,
	),
).to(device)

new_model = neurons.Sequential(
	neurons.LIF(
		num_in=784,
		num_out=num_hidden,
	),
	neurons.LIF(
		num_in=num_hidden,
		num_out=num_hidden,
	),
	neurons.LIS(
		num_in=num_hidden,
		num_out=10,
	),
).to(device)

lr = 1e-4
old_model_optimizer = torch.optim.Adam(old_model.parameters(), lr=lr)
new_model_optimizer = torch.optim.Adam(new_model.parameters(), lr=lr)

old_model_accuracy = tt.plot.MeasurementManager(title="Old Model Accuracy")
new_model_accuracy = tt.plot.MeasurementManager(title="New Model Accuracy")

num_epochs = 1
think_steps = 10

from time import time

old_model_time = 0
new_model_time = 0

for epoch in range(num_epochs):
	for _, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{epoch}"):
		x, y = x.to(device), y.to(device)
		correct_class = y.argmax().item()

		old_model.zero_states()
		aggregate = torch.zeros(old_model.num_out).to(device)
		old_start = time()
		for t in range(think_steps):
			old_out = old_model.forward(torch.bernoulli(x))
			aggregate += old_out
			loss, ls = tt.loss.mse(old_out, y)
			old_model.backward(ls)
		old_model_time += time() - old_start
		old_model_optimizer.step()
		aggregate /= think_steps
		old_model_accuracy.append(1 if aggregate.argmax().item() == correct_class else 0)
		old_model.zero_grad()

		new_model.zero_states()
		aggregate = torch.zeros(new_model.num_out).to(device)
		new_start = time()
		for t in range(think_steps):
			new_out = new_model.forward(torch.bernoulli(x))
			aggregate += new_out
			loss, ls = tt.loss.mse(new_out, y)
			new_model.backward(ls)
		new_model.elig_to_grad()
		new_model_time += time() - new_start
		new_model_optimizer.step()
		aggregate /= think_steps
		new_model_accuracy.append(1 if aggregate.argmax().item() == correct_class else 0)
		new_model.zero_grad()

		if (_ + 1) % 10_000 == 0:
			old_model_accuracy.plot()
			new_model_accuracy.plot()

old_model_accuracy.plot()
new_model_accuracy.plot()

old_model_time /= len(train_dataloader)
new_model_time /= len(train_dataloader)

print(f"AVG Old: {old_model_time:.5f}s for {think_steps} forward / backward passes")
print(f"AVG New: {new_model_time:.5f}s for {think_steps} forward / backward passes")
