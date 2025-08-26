import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn
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

actor = snn.Sequential(
	snn.LIF(
		num_in=784,
		num_out=num_hidden,
	),
	snn.LIF(
		num_in=num_hidden,
		num_out=num_hidden,
	),
	snn.LIS(
		num_in=num_hidden,
		num_out=10,
	),
).to(device)

critic = nn.Sequential(
	nn.LazyLinear(out_features=num_hidden),
	nn.LeakyReLU(),
	nn.LazyLinear(out_features=num_hidden),
	nn.LeakyReLU(),
	nn.LazyLinear(out_features=1)
).to(device)

lr = 1e-5
actor_optimizer = torch.optim.SGD(actor.parameters(), lr=lr)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=lr)

accuracy_manager = tt.plot.MeasurementManager(title="New Model Accuracy")

num_epochs = 1
think_steps = 10

for epoch in range(num_epochs):
	for _, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{epoch}"):
		x, y = x.to(device), y.to(device)
		correct_class = y.argmax().item()

		actor.zero_states()
		actor.zero_grad()
		critic.zero_grad()

		for t in range(think_steps):
			current_state = torch.bernoulli(x)
			value = critic.forward(current_state)
			distribution = actor.forward(current_state)
			sample = tt.functional.sample_softmax(distribution)
			reward = 1 if sample.argmax().item() == correct_class else 0
			next_state = torch.bernoulli(x)
			next_value = critic.forward(next_state)
			actor_loss, actor_ls, critic_loss = tt.loss.actor_critic(distribution, sample, value, next_value, reward)
			critic_loss.backward()
			actor.backward(actor_ls)
			actor.elig_to_grad()
			critic_optimizer.step()
			critic.zero_grad()
			actor_optimizer.step()
			actor.zero_grad()

			accuracy_manager.append(reward)

		if (_ + 1) % 10_000 == 0:
			accuracy_manager.plot()

accuracy_manager.plot()
