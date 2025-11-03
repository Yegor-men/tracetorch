"""
A test on long sequence manipulations, MNIST

The previous MNIST tutorial focused on if the model could learn to generalize to noisy and clean data using various
training methods, such as full BPTT or BPTT truncated at each timestep, thus getting online learning. We tested both
dense and sparse learning signals, and intentionally lowered the decay values to force the model to learn to accumulate
charge. However, this isn't really an RNN model in the sense. We were just training the model to accumulate charge,
not necessarily do anything useful. The order in which we got the signals didn't really matter, they'd just be
accumulated anyway, and there was no need to manipulate the accumulated signals. If we were looking at this as a
function through time, the previous example is monotonic, while this one is nonmonotonic. If the model can learn to
figure out temporal dynamics from a sparse learning signal, we can hope that the model will be able to learn in RL or
other systems of sparse learning signals. It's still imperfect, in that theoretically the model might somehow learn to
represent each timestep and thus reverse engineer the coordinate, and now it's back to being a monotonic function, but
still, it's a lot better than before and at that point you may raise the argument that the goal of any SNN is to turn
a temporally nonmonotonic function into a monotonic one.

The idea here is that the model inspects the image over 784 timesteps, it's basically sliding a 1x1 kernel across the
image. It effectively gets a long chain of 1s and 0s throughout time that represent the image, and it must learn to map
that long sequence to a classification. Unlike the previous example, here, we don't have a per timestep learning signal,
we don't know at what timestep it becomes obvious what the number is, so the model must somehow store all this in
working memory and figure it out.
"""
import time

# ======================================================================================================================

import torch
from torch import nn
import tracetorch as tt
from tracetorch.snn import auto as a

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


# ======================================================================================================================


def one_hot_encode(label):
	return nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotFlattenMNIST(torch.utils.data.Dataset):
	def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0):
		self.dataset = datasets.MNIST(
			root="data",
			train=train,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),  # -> [1,28,28] in [0,1]
				transforms.Lambda(lambda x: x.view(-1)),  # -> [784]
				transforms.Lambda(lambda x: x * (max_val - min_val) + min_val + offset)
			])
		)

	def __getitem__(self, index):
		img, label = self.dataset[index]  # img: [784], label: int
		return img, label

	def __len__(self):
		return len(self.dataset)


def sliding_collate(batch):
	imgs, labels = zip(*batch)
	imgs_b_784 = torch.stack(imgs, dim=0)  # [B, 784]
	imgs_t_b_1 = imgs_b_784.transpose(0, 1).unsqueeze(-1)  # [784, B, 1]
	labels_tensor = torch.tensor(labels, dtype=torch.long)
	labels_onehot = nn.functional.one_hot(labels_tensor, num_classes=10).float()  # [B, 10]
	return imgs_t_b_1, labels_onehot


# ======================================================================================================================

min_prob, max_prob, noise_offset = 0.0, 1.0, 0.0

train_dataset = OneHotFlattenMNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = OneHotFlattenMNIST(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

batch_size = 100

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sliding_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=sliding_collate)

# ======================================================================================================================

base_beta_decay = tt.functional.halflife_to_decay(784)  # 784 total timesteps, want the signals to still be kept alive
print(f"Base Beta: {base_beta_decay}")

hidden_features = 128
model_template = a.Sequential(
	nn.Linear(1, hidden_features),
	a.RLeaky(hidden_features, base_beta_decay),
	nn.Linear(hidden_features, hidden_features),
	a.RLeaky(hidden_features, base_beta_decay),
	nn.Linear(hidden_features, hidden_features),
	a.RLeaky(hidden_features, base_beta_decay),
	nn.Linear(hidden_features, 10),
	a.Readout(10, base_beta_decay),
	nn.Softmax(-1)
).to(device=device)

num_epochs = 1

total_params = sum(p.numel() for p in model_template.parameters())
print(f"Total parameters: {total_params:,}")
time.sleep(0.1)

loss_fn = tt.loss.soft_cross_entropy
# loss_fn = nn.functional.mse_loss

train_types = ["bptt", "online"]
for train_type in train_types:
	model = copy.deepcopy(model_template)
	optimizer = torch.optim.AdamW(model.parameters(), 5e-5)

	loss_manager = tt.plot.MeasurementManager(f"{train_type} - Loss", [0.0, 0.9, 0.99])
	accuracy_manager = tt.plot.MeasurementManager(f"{train_type} - Accuracy", [0.0, 0.9, 0.99])

	# TRAIN
	model.train()
	for e in range(num_epochs):
		for image_seq, label in tqdm(train_dataloader, total=len(train_dataloader), desc=f"{train_type}, E{e} - TRAIN"):
			image_seq, label = torch.bernoulli(image_seq).to(device=device), label.to(device=device)
			t, b, s = image_seq.shape

			model.zero_states()
			model.zero_grad()

			if train_type == "bptt":
				for image in image_seq:
					model_output = model(image)
				loss = loss_fn(model_output, label)
				loss.backward()

			if train_type == "online":
				for index, image in enumerate(image_seq):
					if index < t - 1:
						with torch.no_grad():
							model_output = model(image)
					else:
						model_output = model(image)
				loss = loss_fn(model_output, label)
				loss.backward()

			optimizer.step()

			loss_manager.append(loss.item())
			pred_classes = model_output.argmax(dim=1)
			true_classes = label.argmax(dim=1)
			num_correct = (pred_classes == true_classes).sum().item() / b
			accuracy_manager.append(num_correct)

	loss_manager.plot()
	accuracy_manager.plot()

	# TEST
	model.eval()
	test_loss, test_accuracy = 0, 0
	for image_seq, label in tqdm(test_dataloader, total=len(test_dataloader), desc=f"{train_type} - TEST"):
		with torch.no_grad():
			image_seq, label = torch.bernoulli(image_seq).to(device=device), label.to(device=device)
			t, b, s = image_seq.shape

			model.zero_states()

			for image in image_seq:
				model_output = model(image)
			loss = loss_fn(model_output, label)

			test_loss += loss.item()
			true_classes = label.argmax(dim=1)
			pred_classes = model_output.argmax(dim=1)
			correct = (pred_classes == true_classes).sum().item()
			test_accuracy += correct

		test_loss /= len(test_dataloader)
		test_accuracy /= len(test_dataloader.dataset)

	# PRINT
	print(f"Test - Loss: {test_loss:.5f}, Accuracy: {test_accuracy * 100:.5f}%")

# VISUALIZE
# with torch.no_grad():
# 	images, labels = next(iter(test_dloader))
# 	images, labels = images.to(device), labels.to(device)
#
# 	for i, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(images), desc=f"{tr_type} - VISUALIZE"):
# 		if i >= num_to_visualize:
# 			continue
#
# 		empty_image = torch.zeros_like(image)
#
# 		spike_train = []
# 		model.zero_states()
#
# 		for t in range(num_timesteps):
# 			spike_input = torch.bernoulli(image)
# 			model_output = model(spike_input.unsqueeze(0)).squeeze()
# 			empty_image += spike_input
# 			spike_train.append(model_output.squeeze())
#
# 		empty_image /= num_timesteps
# 		loss = loss_fn(model_output, label)
# 		real_number = torch.argmax(label).item()
# 		pred_number = torch.argmax(model_output).item()
# 		title = f"{tr_type} - Was: {real_number}, pred: {pred_number}, loss: {loss.item():.5f}"
# 		tt.plot.render_image(empty_image, title=title)
# 		tt.plot.spike_train(spike_train, title=title)
