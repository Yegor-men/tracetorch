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

num_epochs = 3
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

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for e in range(num_epochs):
	# TRAIN
	train_loss_manager = tt.plot.MeasurementManager(f"Train loss, epoch {e + 1}")
	train_loss = 0
	train_correct = 0
	for (image, label) in tqdm(train_dloader, total=len(train_dloader), desc=f"Epoch {e + 1}, train"):
		image, label = image.to(device), label.to(device)

		model.zero_states()
		model.zero_grad()

		for t in range(num_timesteps):
			spike_input = torch.bernoulli(image)
			model_output = model(spike_input)

		pred_classes = model_output.argmax(dim=1)
		true_classes = label.argmax(dim=1)
		correct = (pred_classes == true_classes).sum().item()
		train_correct += correct

		loss = nn.functional.mse_loss(model_output, label)
		train_loss += loss.item()
		train_loss_manager.append(loss.item())
		loss.backward()
		optimizer.step()

	train_correct /= 60_000
	train_accuracies.append(train_correct)
	train_loss /= len(train_dloader)
	train_losses.append(train_loss)
	train_loss_manager.plot()

	# TEST
	test_loss = 0
	test_correct = 0
	for (image, label) in tqdm(test_dloader, total=len(test_dloader), desc=f"Epoch {e + 1}, test"):
		with torch.no_grad():
			image, label = image.to(device), label.to(device)

			model.zero_states()

			for t in range(num_timesteps):
				spike_input = torch.bernoulli(image)
				model_output = model(spike_input)

			pred_classes = model_output.argmax(dim=1)
			true_classes = label.argmax(dim=1)
			correct = (pred_classes == true_classes).sum().item()
			test_correct += correct

			loss = nn.functional.mse_loss(model_output, label)
			test_loss += loss.item()

	test_correct /= 10_000
	test_accuracies.append(test_correct)
	test_loss /= len(test_dloader)
	test_losses.append(test_loss)

	# PLOT
	print(f"Train - Loss: {train_loss:.5f}, Accuracy: {train_accuracies[-1] * 100:.5f}%")
	print(f"Test - Loss: {test_loss:.5f}, Accuracy: {test_accuracies[-1] * 100:.5f}%")

	plt.plot(train_losses, label="Train")
	plt.plot(test_losses, label="Test")
	plt.title("Loss")
	plt.legend()
	plt.show()

	plt.plot(train_accuracies, label="Train")
	plt.plot(test_accuracies, label="Test")
	plt.title("Accuracy")
	plt.legend()
	plt.show()

	# VISUALIZE
	with torch.no_grad():
		images, labels = next(iter(test_dloader))
		images, labels = images.to(device), labels.to(device)

		for image, label in tqdm(zip(images, labels), total=len(images), desc=f"Epoch {e + 1}, visualization"):
			spike_train = []
			model.zero_states()

			for t in range(num_timesteps):
				model_output = model(image.unsqueeze(0)).squeeze()
				spike_train.append(model_output.squeeze())

			loss = nn.functional.mse_loss(model_output, label)
			tt.plot.render_image(image, title=f"{torch.argmax(label).item()}")
			tt.plot.spike_train(spike_train, title=f"{loss.item()}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
