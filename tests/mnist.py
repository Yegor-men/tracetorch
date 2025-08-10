import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# Parameter: scale factor for pixel values
alpha = 1  # Default alpha; set to any value in [0,1] as needed
beta = 0.05

# Image transforms: convert to tensor, flatten, scale by alpha
image_transform = transforms.Compose([
	transforms.ToTensor(),  # [0,1] conversion
	transforms.Lambda(lambda x: x.view(-1)),  # flatten from [1,28,28] to [784]
	transforms.Lambda(lambda x: x * (alpha - beta) + beta)  # scale to [0, alpha]
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
		weight_scaling=0.05,
		config=config_dict,
	),
	tracetorch.nn.LIF(
		n_in=128,
		n_out=128,
		weight_scaling=0.05,
		config=config_dict,
	),
	tracetorch.nn.LIF(
		n_in=128,
		n_out=128,
		weight_scaling=0.05,
		config=config_dict,
	),
	tracetorch.nn.Softmax(
		n_in=128,
		n_out=10,
		weight_scaling=0.05,
		config=config_dict,
	),
)

n_epochs = 1
n_think = 10

decay = torch.tensor([0.9, 0.99, 0.999]).to(config_dict["device"])
train_loss_manager = tracetorch.plot.MeasurementManager(title="Train Loss", decay=decay)
train_acc_manager = tracetorch.plot.MeasurementManager(title="Train Accuracy", decay=decay)

num_train_samples = len(train_loader)

for epoch in range(n_epochs):
	for index, (x, y) in tqdm(enumerate(train_loader), total=num_train_samples, leave=True, desc=f"Epoch {epoch + 1}"):
		model.zero_states()
		aggregate = torch.zeros_like(model.layers[-1].mem)
		for i in range(n_think):
			model_out = model.forward(torch.bernoulli(x))
			aggregate += model_out
		aggregate /= aggregate.sum()
		loss, ls = tracetorch.loss.mse(torch.nn.functional.softmax(model.layers[-1].mem, dim=-1), y)
		model.backward(ls)
		train_loss_manager.append(loss)
		train_acc_manager.append(1 if aggregate.argmax().item() == y.argmax().item() else 0)
		if index % 10_000 == 0:
			train_loss_manager.plot(title=f"Loss: Epoch {epoch}, image {index}")
			train_acc_manager.plot(title=f"Accuracy: Epoch {epoch}, image {index}")
			trace_decays = [torch.nn.functional.sigmoid(layer.in_trace_decay) for layer in model.layers]
			tracetorch.plot.render_image(trace_decays[0].view(28, 28), title=f"L1 decay: E {epoch}, I {index}")
			tracetorch.plot.distributions(f"Input trace decays: E {epoch}, I {index}", trace_decays)

train_loss_manager.plot(title=f"Loss, training finished")
train_acc_manager.plot(title=f"Accuracy, training finished")
trace_decays = [torch.nn.functional.sigmoid(layer.in_trace_decay) for layer in model.layers]
tracetorch.plot.render_image(trace_decays[0].view(28, 28), title=f"In trace decay, training finished")
tracetorch.plot.distributions("Input trace decays training finished", trace_decays)

test_loss_manager = tracetorch.plot.MeasurementManager(title="Test Loss", decay=decay)
num_test_samples = len(test_loader)
correct_test_predictions = 0

for index, (x, y) in tqdm(enumerate(test_loader), total=num_test_samples, leave=True, desc=f"Testing"):
	model.zero_states()
	aggregate = torch.zeros_like(model.layers[-1].mem)
	for i in range(n_think):
		model_out = model.forward(torch.bernoulli(x))
		aggregate += model_out
	aggregate /= aggregate.sum()
	loss, ls = tracetorch.loss.mse(torch.nn.functional.softmax(model.layers[-1].mem, dim=-1), y)
	test_loss_manager.append(loss)
	if aggregate.argmax().item() == y.argmax().item():
		correct_test_predictions += 1
accuracy = correct_test_predictions / num_test_samples
test_loss_manager.plot(title=f"Test accuracy: {(accuracy * 100):.2f}%")
