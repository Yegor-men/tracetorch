import torch
import tracetorch as tt
from tracetorch import snn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

min_prob = 0.05
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

n_hidden = 128

model = snn.Sequential(
	snn.LIF(
		num_in=784,
		num_out=n_hidden,
	),
	snn.LIF(
		num_in=n_hidden,
		num_out=n_hidden,
	),
	snn.LIS(
		num_in=n_hidden,
		num_out=10,
	)
).to(device)

think_steps = 5
num_epochs = 1

from tqdm import tqdm

loss_manager = tt.plot.MeasurementManager(title="Loss")
accuracy_manager = tt.plot.MeasurementManager(title="Accuracy")

optimizer = torch.optim.Adam(params=model.get_learnable_parameters(), lr=1e-5)

for epoch in range(num_epochs):
	for index, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{epoch}"):
		model.clear_grad()
		model.zero_states()
		for _ in range(think_steps):
			model_dist = model.forward(torch.bernoulli(x).to(device))
		loss, ls = tt.loss.mse(model_dist, y.to(device))
		model.backward(ls)
		optimizer.step()
		loss_manager.append(loss)
		accuracy_manager.append(1 if model_dist.argmax().item() == y.argmax().item() else 0)
		if index % 10000 == 0:
			loss_manager.plot()
			accuracy_manager.plot()

loss_manager.plot()
accuracy_manager.plot()

net_loss = torch.zeros_like(loss)
net_accuracy = 0

for index, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True, desc=f"Testing"):
	model.zero_states()
	for _ in range(think_steps):
		model_dist = model.forward(torch.bernoulli(x).to(device))
	loss, ls = tt.loss.mse(model_dist, y.to(device))
	net_loss += loss
	net_accuracy += (1 if model_dist.argmax().item() == y.argmax().item() else 0)

net_loss /= len(test_dataloader)
net_accuracy /= len(test_dataloader)

print(f"\tTEST\nLoss: {net_loss}\nAccuracy: {net_accuracy}")

in_trace_decays = [torch.nn.functional.sigmoid(layer.in_trace_decay) for layer in model.layers]
mem_decays = [torch.nn.functional.sigmoid(layer.mem_decay) for layer in model.layers]
weights = [layer.weight for layer in model.layers]
thresholds = [torch.nn.functional.softplus(layer.threshold) for layer in model.layers[:-1]]

tt.plot.distributions(in_trace_decays, title="Input trace decay")
tt.plot.distributions(mem_decays, title="Membrane decay")
tt.plot.distributions(weights, title="Weights")
tt.plot.distributions(thresholds, title="Thresholds")
