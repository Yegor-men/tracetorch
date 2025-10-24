import torch
from torch import nn

import tracetorch as tt
from tracetorch import snn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt


# ======================================================================================================================

def one_hot_encode(label):
	return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNIST(torch.utils.data.Dataset):
	def __init__(self, train=True):
		self.dataset = datasets.MNIST(
			root='data',
			train=train,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),  # Converts to [C, H, W] in [0.0, 1.0]
			])
		)

	def __getitem__(self, index):
		image, label = self.dataset[index]
		one_hot_label = one_hot_encode(label)
		return image, one_hot_label

	def __len__(self):
		return len(self.dataset)


train_dataset = OneHotMNIST(train=True)
test_dataset = OneHotMNIST(train=False)

# ======================================================================================================================

num_epochs = 1
num_timesteps = 50
batch_size = 32

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = snn.Sequential(
	nn.Conv2d(1, 8, 3, dilation=2),
	snn.Leaky(8, 0.9, 1.0, view_tuple=(1, -1, 1, 1)),
	nn.Conv2d(8, 16, 3, dilation=2),
	snn.Leaky(16, 0.9, 1.0, view_tuple=(1, -1, 1, 1)),
	nn.Conv2d(16, 32, 3, dilation=2),
	nn.Flatten(),
	nn.LazyLinear(10),
	snn.Readout(10, 0.9),
	nn.Softmax(-1),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
loss_manager = tt.plot.MeasurementManager("Loss", [0.5, 0.9, 0.99])
accuracy_manager = tt.plot.MeasurementManager("Accuracy", [0.5, 0.9, 0.99])

for e in range(num_epochs):
	# TRAIN
	temp_loss, temp_accuracy = 0, 0
	for (image, label) in tqdm(train_dloader, total=len(train_dloader), desc=f"E{e + 1} - TRAIN"):
		b, c, h, w = image.shape
		image, label = image.to(device), label.to(device)

		model.zero_states()
		model.zero_grad()

		for t in range(num_timesteps):
			spike_input = torch.bernoulli(image)

			model_output = model(spike_input)

			loss = nn.functional.mse_loss(model_output, label)
			# loss = torch.mean(torch.log(model_output) * label)
			loss.backward()
			model.detach_states()

		with torch.no_grad():
			loss_manager.append(loss.item())
			temp_loss += loss.item()
			true_classes = label.argmax(dim=1)
			pred_classes = model_output.argmax(dim=1)
			correct = (pred_classes == true_classes).sum().item()
			temp_accuracy += correct
			correct /= b
			accuracy_manager.append(correct)

			for param in model.parameters():
				param.grad.div_(num_timesteps)

		optimizer.step()

	temp_loss /= len(train_dloader)
	temp_accuracy /= 60_000
	train_losses.append(temp_loss)
	train_accuracies.append(temp_accuracy)

	loss_manager.plot()
	accuracy_manager.plot()

	# TEST
	temp_loss, temp_accuracy = 0, 0
	with torch.no_grad():
		for (image, label) in tqdm(test_dloader, total=len(test_dloader), desc=f"E{e + 1} - TEST"):
			b, c, h, w = image.shape
			image, label = image.to(device), label.to(device)

			model.zero_states()
			model.zero_grad()

			for t in range(num_timesteps):
				spike_input = torch.bernoulli(image)
				model_output = model(spike_input)
				loss = nn.functional.mse_loss(model_output, label)
			# loss = torch.mean(torch.log(model_output) * label)

			temp_loss += loss.item()
			true_classes = label.argmax(dim=1)
			pred_classes = model_output.argmax(dim=1)
			correct = (pred_classes == true_classes).sum().item()
			temp_accuracy += correct

		temp_loss /= len(test_dloader)
		temp_accuracy /= 10_000
		test_losses.append(temp_loss)
		test_accuracies.append(temp_accuracy)

	# PLOT
	print(f"Train - Loss: {train_losses[-1]:.5f}, Accuracy: {train_accuracies[-1] * 100:.5f}%")
	print(f"Test - Loss: {test_losses[-1]:.5f}, Accuracy: {test_accuracies[-1] * 100:.5f}%")

	plt.title("LOSS")
	plt.plot(train_losses, label="Train")
	plt.plot(test_losses, label="Test")
	plt.legend()
	plt.show()

	plt.title("ACCURACY")
	plt.plot(train_accuracies, label="Train")
	plt.plot(test_accuracies, label="Test")
	plt.legend()
	plt.show()

	# VISUALIZE
	with torch.no_grad():
		images, labels = next(iter(test_dloader))
		images, labels = images.to(device), labels.to(device)

		for image, label in tqdm(zip(images, labels), total=len(images), desc=f"E{e + 1} - VISUALIZE"):
			spike_train = []
			model.zero_states()

			for t in range(num_timesteps):
				model_output = model(image.unsqueeze(0)).squeeze()
				spike_train.append(model_output.squeeze())

			loss = nn.functional.mse_loss(model_output, label)
			# loss = torch.mean(torch.log(model_output) * label)
			real_number = torch.argmax(label).item()
			pred_number = torch.argmax(model_output).item()
			title = f"Was: {real_number}, pred: {pred_number}, loss: {loss.item():.5f}"
			tt.plot.render_image(image, title=title)
			tt.plot.spike_train(spike_train, title=title)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
