import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# Parameter: scale factor for pixel values
alpha = 1.0  # Default alpha; set to any value in [0,1] as needed

# Image transforms: convert to tensor, flatten, scale by alpha
image_transform = transforms.Compose([
	transforms.ToTensor(),  # [0,1] conversion
	transforms.Lambda(lambda x: x.view(-1)),  # flatten from [1,28,28] to [784]
	transforms.Lambda(lambda x: x * alpha)  # scale to [0, alpha]
])


# Target transform: one-hot encode labels into 10-dimensional tensor

def one_hot_encode(y):
	# y is a scalar label (0-9)
	return F.one_hot(torch.tensor(y), num_classes=10).float()


# Custom collate function: remove batch dimension so each sample is returned as tensors [784] and [10]
def collate_fn(batch):
	image, label = batch[0]  # batch is a list of length 1
	return image.squeeze(), label.squeeze()


# Download and prepare MNIST datasets with the above transforms
dataset_kwargs = dict(root='data', download=True,
					  transform=image_transform,
					  target_transform=one_hot_encode)
train_dataset = datasets.MNIST(train=True, **dataset_kwargs)
test_dataset = datasets.MNIST(train=False, **dataset_kwargs)

# DataLoaders with batch_size=1 and no batching (via collate_fn)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Example: retrieve a single (x, y) sample
# x, y = next(iter(train_loader))
# print(f"x shape: {x}, y shape: {y}")  # x: [784], y: [10]

import tracetorch
from tqdm import tqdm

config_dict = {
	"device": "cuda",
	"lr": 1e-3,
}

model = tracetorch.nn.Sequential(
	tracetorch.nn.LIF(
		n_in=784,
		n_out=128,
		config=config_dict,
	),
	tracetorch.nn.LIF(
		n_in=128,
		n_out=128,
		config=config_dict,
	),
	tracetorch.nn.Softmax(
		n_in=128,
		n_out=10,
		config=config_dict,
	),
)

n_epochs = 1
n_think = 10

train_loss = []
train_accuracy = []
num_train_samples = len(train_loader)

for epoch in range(n_epochs):
	epoch_loss = torch.tensor(0.).to(config_dict["device"])
	correct_train_predictions = 0
	for index, (x, y) in tqdm(enumerate(train_loader), total=num_train_samples, leave=True, desc=f"Epoch {epoch + 1}"):
		model.zero_states()
		aggregate = torch.zeros_like(model.layers[-1].mem)
		for i in range(n_think):
			model_out = model.forward(x)
			aggregate += model_out
		aggregate /= aggregate.sum()
		loss, ls = tracetorch.loss.mse(aggregate, y)
		model.backward(ls)
		epoch_loss += loss
		if aggregate.argmax().item() == y.argmax().item():
			correct_train_predictions += 1
	epoch_loss /= num_train_samples
	train_loss.append(epoch_loss)
	accuracy = correct_train_predictions / num_train_samples
	train_accuracy.append(accuracy)
	print(f"Loss: {epoch_loss}, Accuracy: {(accuracy * 100):.2f}%")
	tracetorch.plot.line_graph(train_loss, title=f"Epoch {epoch + 1} - Loss")
	tracetorch.plot.line_graph(train_accuracy, title=f"Epoch {epoch + 1} - Accuracy")

test_loss = []
num_test_samples = len(test_loader)
correct_test_predictions = 0

for index, (x, y) in tqdm(enumerate(test_loader), total=num_test_samples, leave=True, desc=f"Testing"):
	model.zero_states()
	aggregate = torch.zeros_like(model.layers[-1].mem)
	for i in range(n_think):
		model_out = model.forward(x)
		aggregate += model_out
	aggregate /= aggregate.sum()
	loss, ls = tracetorch.loss.mse(aggregate, y)
	test_loss.append(loss)
	if aggregate.argmax().item() == y.argmax().item():
		correct_test_predictions += 1
accuracy = correct_test_predictions / num_test_samples
tracetorch.plot.line_graph(test_loss,
						   title=f"Loss graph, Accuracy: {(accuracy * 100):.2f}% for {num_test_samples} samples")
