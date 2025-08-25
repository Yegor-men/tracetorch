import torch
import tracetorch as tt
from tracetorch import eprop
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

# ----------------------------------------------------------------------------------------------------------------------

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

n_hidden = 32

decay = tt.functional.halflife_to_decay(5)

model = eprop.Sequential(
	eprop.ALIF(
		num_in=784,
		num_out=n_hidden,
	),
	eprop.ALIF(
		num_in=n_hidden,
		num_out=n_hidden,
	),
	eprop.LIS(
		num_in=n_hidden,
		num_out=10,
	),
).to(device)

think_steps = 10
num_epochs = 1

from tqdm import tqdm

loss_manager = tt.plot.MeasurementManager(title="Loss")
accuracy_manager = tt.plot.MeasurementManager(title="Accuracy")

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
	for index, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{epoch}"):
		x, y = x.to(device), y.to(device)
		correct_class = y.argmax().item()

		model.zero_states()
		model.zero_grad()
		aggregate = torch.zeros(model.num_out).to(device)

		for _ in range(think_steps):
			bern = torch.bernoulli(x)
			model_dist = model.forward(bern)
			aggregate += model_dist

			loss, ls = tt.loss.mse(model_dist, y)
			model.backward(ls)
			loss_manager.append(loss)

		model.elig_to_grad()
		optimizer.step()
		aggregate /= think_steps
		accuracy_manager.append(1 if aggregate.argmax().item() == correct_class else 0)

		if (index % 10000 == 0) and (index != 0):
			loss_manager.plot()
			accuracy_manager.plot()

loss_manager.plot()
accuracy_manager.plot()

net_loss = torch.tensor(0.).to(device)
net_accuracy = 0
mistakes_matrix = torch.zeros(10, 10)

for index, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True, desc=f"Testing"):
	x, y = x.to(device), y.to(device)
	correct_class = y.argmax().item()

	model.zero_states()
	aggregate = torch.zeros(model.num_out).to(device)
	raw_image = torch.zeros_like(x)
	model_dists = []

	for _ in range(think_steps):
		bern = torch.bernoulli(x)
		raw_image += bern
		model_dist = model.forward(bern)
		loss, ls = tt.loss.mse(model_dist, y)
		net_loss += loss

		aggregate += model_dist
		model_dists.append(model_dist)

	aggregate /= aggregate.sum()
	raw_image /= think_steps

	if index % 1000 == 0:
		tt.plot.render_image(raw_image.unsqueeze(0).view(28, 28))
		tt.plot.spike_train(model_dists, title="Model outputs")

	chosen_index = int(aggregate.argmax().item())
	correct = chosen_index == correct_class
	if not correct:
		mistakes_matrix[correct_class][chosen_index] += 1
	else:
		net_accuracy += 1

net_loss /= len(test_dataloader)
net_accuracy /= len(test_dataloader)

print(f"\tTEST\nLoss: {net_loss}\nAccuracy: {net_accuracy}")

tt.plot.render_image(mistakes_matrix)

# mem_decays = [torch.nn.functional.sigmoid(layer.mem_decay) for layer in model.layers[:-1]]
# weights = [layer.weight for layer in model.layers]
# thresholds = [torch.nn.functional.softplus(layer.threshold) for layer in model.layers[:-1]]
#
# tt.plot.distributions(mem_decays, title="Membrane decay")
# tt.plot.distributions(weights, title="Weights")
# tt.plot.distributions(thresholds, title="Thresholds")
