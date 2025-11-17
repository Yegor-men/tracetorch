"""
A test on generalization, MNIST

Training is tested on:
	1. BPTT dense
	2. BPTT sparse
	3. Online dense
	4. Online sparse

The naming convention is a little arbitrary, but the point is that this is a comparison between:
	1. BPTT vs online loss
	2. Dense vs sparse learning signals

Here's what actually happens for the losses and the auto graphs for their construction:
	1. BPTT dense creates a singular loss that is a mix of the loss of all timesteps; it can be set to make all timesteps
	of equal priority, but for the sake of demonstration, we use an EMA for it so that later timesteps are of higher
	priority than earlier ones.
	2. BPTT sparse also creates an auto graph that spans all timesteps, except we ignore all outputs except for the
	last timestep's; this is equivalent to BPTT dense, except setting the EMA decay to 0.0.
	3. Online dense is similar to BPTT dense in that we have a learning signal at each timestep, but now we detach each
	timestep from the graph; instead of having one massive graph to call .backward() on, we call on an arbitrary amount
	of small graphs. We still apply the EMA here, but applying EMA to the loss directly doesn't make much sense, it's
	instead applied to the .grad of the parameters.
	4. Online sparse is equivalent to Online dense, except we ignore all losses except the last timestep's, yet again
	equivalent to if the EMA decay was set to 0.0.

Methods 1 and 3 are for when we explicitly want each at timestep to produce the correct output. Methods 2 and 4 are for
when we don't particularly care about the intermediate outputs (or don't know what it should be), only what the last one
must be. The decay value for 1 and 3 is set so that the halflife is half the number of steps.

We also intentionally make the task more difficult by having a baseline noise to the images, which makes all pixels more
likely to fire, no matter their original color; and we also lower the max brightness so that the brightest pixels don't
have a 100% chance of firing. The data is thus made to be much noisier; the brightest pixels should fire on average only
once throughout the entire spike train. We also set the initial beta value for the model (membrane decay) intentionally
low so that the model will lose information; if beta was high like 0.999, with minimal effort the model would just be
able to cheat and accumulate the charge throughout the spike train; the model must learn to accumulate charge.

The rendered images may be hard to identify; that is the intended effect, that is what the model optimally sees assuming
it had learnt to maximally accumulate charge.
"""

# ======================================================================================================================


import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================================================================

def one_hot_encode(label):
	return torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNIST(torch.utils.data.Dataset):
	def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0):
		self.dataset = datasets.MNIST(
			root='data',
			train=train,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				lambda x: x * (max_val - min_val) + min_val + offset,
			])
		)

	def __getitem__(self, index):
		image, label = self.dataset[index]
		one_hot_label = one_hot_encode(label)
		return image, one_hot_label

	def __len__(self):
		return len(self.dataset)


# ======================================================================================================================

num_epochs = 3
num_timesteps = 20
ema_decay = tt.functional.halflife_to_decay(num_timesteps / 2)
model_layers_beta = tt.functional.halflife_to_decay(num_timesteps / 100)
print(f"Original Decay: {model_layers_beta}")

noise_offset = 0.1
min_prob = 0.0
max_prob = (1.0 / num_timesteps)

# min_prob, max_prob, noise_offset = 0.0, 0.05, 0.01

train_dataset = OneHotMNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = OneHotMNIST(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

batch_size = 32
num_to_visualize = 5

train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ======================================================================================================================


model_template = snn.Sequential(
	nn.Conv2d(1, 8, 5, dilation=2),
	# kernel 5x5 @ dilation 2 becomes 9x9, 28-9+1=20
	snn.Leaky(8, model_layers_beta, 1.0, view_tuple=(1, -1, 1, 1)),
	nn.Conv2d(8, 16, 5, dilation=2),
	# kernel 5x5 @ dilation 2 becomes 9x9, 20-9+1 = 12
	snn.Leaky(16, model_layers_beta, 1.0, view_tuple=(1, -1, 1, 1)),
	nn.Conv2d(16, 16, 5, dilation=2),
	# kernel 5x5 @ dilation 2 becomes 9x9, 12-9+1 = 4
	nn.Flatten(),
	# flatten preserves the 0th dim, so must keep batch, that's why earlier layers view tuple includes it
	nn.Linear(256, 10),  # 4x4x16=256
	snn.Readout(10, model_layers_beta),
	nn.Softmax(-1),
).to(device)

total_params = sum(p.numel() for p in model_template.parameters())
print(f"Total parameters: {total_params:,}")

loss_fn = tt.loss.soft_cross_entropy
# loss_fn = nn.functional.mse_loss


train_types = ["bptt_dense", "bptt_sparse", "online_dense", "online_sparse"]

for tr_type in train_types:
	model = copy.deepcopy(model_template)

	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

	loss_manager = tt.plot.MeasurementManager(f"{tr_type} - Loss", [0.0, 0.9, 0.99])
	accuracy_manager = tt.plot.MeasurementManager(f"{tr_type} - Accuracy", [0.0, 0.9, 0.99])

	# TRAIN
	model.train()
	for e in range(num_epochs):
		for (image, label) in tqdm(train_dloader, total=len(train_dloader), desc=f"{tr_type} - E{e + 1} - TRAIN"):
			b, c, h, w = image.shape
			image, label = image.to(device), label.to(device)

			model.zero_states()
			model.zero_grad()

			if tr_type == "bptt_dense":
				running_loss = torch.tensor(0, device=device)
				for t in range(num_timesteps):
					spike_input = torch.bernoulli(image)
					model_output = model(spike_input)
					loss = loss_fn(model_output, label)
					running_loss = running_loss * ema_decay + loss * (1 - ema_decay)

				recording_loss = running_loss.item()
				running_loss.backward()

			if tr_type == "bptt_sparse":
				running_loss = torch.tensor(0, device=device)
				for t in range(num_timesteps):
					spike_input = torch.bernoulli(image)
					model_output = model(spike_input)
					loss = loss_fn(model_output, label)
					running_loss = running_loss * 0.0 + loss

				recording_loss = running_loss.item()
				running_loss.backward()

			if tr_type == "online_dense":
				for t in range(num_timesteps):
					with torch.no_grad():
						for param in model.parameters():
							if param.grad is not None:
								param.grad.mul_(ema_decay)

					spike_input = torch.bernoulli(image)
					model_output = model(spike_input)
					loss = loss_fn(model_output, label) * (1 - ema_decay)
					recording_loss = loss.item()
					loss.backward()
					model.detach_states()

			if tr_type == "online_sparse":
				for t in range(num_timesteps):
					with torch.no_grad():
						for param in model.parameters():
							if param.grad is not None:
								param.grad.mul_(0.0)

					spike_input = torch.bernoulli(image)
					model_output = model(spike_input)
					loss = loss_fn(model_output, label)
					recording_loss = loss.item()
					loss.backward()
					model.detach_states()

			optimizer.step()
			with torch.no_grad():
				loss_manager.append(recording_loss)
				pred_classes = model_output.argmax(dim=1)
				true_classes = label.argmax(dim=1)
				num_correct = (pred_classes == true_classes).sum().item() / b
				accuracy_manager.append(num_correct)

	loss_manager.plot()
	accuracy_manager.plot()

	# TEST
	model.eval()
	test_loss, test_accuracy = 0, 0
	with torch.no_grad():
		for (image, label) in tqdm(test_dloader, total=len(test_dloader), desc=f"{tr_type} - TEST"):
			b, c, h, w = image.shape
			image, label = image.to(device), label.to(device)

			model.zero_states()
			model.zero_grad()

			for t in range(num_timesteps):
				spike_input = torch.bernoulli(image)
				model_output = model(spike_input)
				loss = loss_fn(model_output, label)

			test_loss += loss.item()
			true_classes = label.argmax(dim=1)
			pred_classes = model_output.argmax(dim=1)
			correct = (pred_classes == true_classes).sum().item()
			test_accuracy += correct

		test_loss /= len(test_dloader)
		test_accuracy /= len(test_dloader.dataset)

	# PRINT
	print(f"Test - Loss: {test_loss:.5f}, Accuracy: {test_accuracy * 100:.5f}%")

	model_betas = model.get_attr_list("beta", "beta_scalar")
	for t in model_betas:
		beta = nn.functional.sigmoid(t)
		print(f"Beta - mean: {beta.mean()}, std: {beta.std() if beta.numel() > 1 else None}")

	# VISUALIZE
	with torch.no_grad():
		images, labels = next(iter(test_dloader))
		images, labels = images.to(device), labels.to(device)

		for i, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(images), desc=f"{tr_type} - VISUALIZE"):
			if i >= num_to_visualize:
				continue

			empty_image = torch.zeros_like(image)

			spike_train = []
			model.zero_states()

			for t in range(num_timesteps):
				spike_input = torch.bernoulli(image)
				model_output = model(spike_input.unsqueeze(0)).squeeze()
				empty_image += spike_input
				spike_train.append(model_output.squeeze())

			empty_image /= num_timesteps
			loss = loss_fn(model_output, label)
			real_number = torch.argmax(label).item()
			pred_number = torch.argmax(model_output).item()
			title = f"{tr_type} - Was: {real_number}, pred: {pred_number}, loss: {loss.item():.5f}"
			tt.plot.render_image(empty_image, title=title)
			tt.plot.spike_train(spike_train, title=title)
