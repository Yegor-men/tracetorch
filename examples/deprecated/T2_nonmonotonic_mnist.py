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
from tracetorch import snn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# ======================================================================================================================


min_prob, max_prob, noise_offset = 0.0, 1.0, 0.0
batch_size = 64
kernel_size = 4
stride = 4
pad = False
num_workers = 2
pin_memory = True


def one_hot_encode(label):
	return nn.functional.one_hot(torch.tensor(label), num_classes=10).float()


class OneHotMNISTImage(torch.utils.data.Dataset):
	def __init__(self, train=True, min_val=0.0, max_val=1.0, offset=0.0, root="data"):
		self.ds = datasets.MNIST(
			root=root,
			train=train,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),  # -> [C, H, W] in [0,1]
				transforms.Lambda(lambda x: x * (max_val - min_val) + min_val + offset)
			])
		)

	def __len__(self):
		return len(self.ds)

	def __getitem__(self, idx):
		img, label = self.ds[idx]  # img: [C, H, W], label: int
		return img, label


def patch_collate(batch):
	"""
	returns:
	  imgs_orig:  [B, C, H, W]
	  seq:        [T, B, area]  (area = C * k * k)
	  labels_onehot: [B, 10]
	"""
	imgs, labels = zip(*batch)
	imgs_b = torch.stack(imgs, dim=0)  # [B, C, H, W]
	B, C, H, W = imgs_b.shape

	# optional pad so patches tile exactly
	if pad:
		rem_h = (H - kernel_size) % stride
		rem_w = (W - kernel_size) % stride
		pad_h = (stride - rem_h) % stride
		pad_w = (stride - rem_w) % stride
		if pad_h != 0 or pad_w != 0:
			imgs_b = nn.functional.pad(imgs_b, (0, pad_w, 0, pad_h), value=0.0)  # pad (left,right,top,bottom) order
			_, _, H, W = imgs_b.shape

	# unfold to patches: [B, C, n_h, n_w, k, k]
	patches = imgs_b.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
	# permute and reshape to [B, n_patches, area]
	patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
	n_h = patches.size(1)
	n_w = patches.size(2)
	n_patches = n_h * n_w
	patches = patches.view(B, n_patches, C * kernel_size * kernel_size)  # [B, T, area]
	# transpose to [T, B, area]
	seq = patches.permute(1, 0, 2).contiguous()  # [T, B, area]

	labels_tensor = torch.tensor(labels, dtype=torch.long)
	labels_onehot = nn.functional.one_hot(labels_tensor, num_classes=10).float()  # [B, 10]

	return imgs_b, seq, labels_onehot


train_dataset = OneHotMNISTImage(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = OneHotMNISTImage(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
							  collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
							 collate_fn=patch_collate, num_workers=num_workers, pin_memory=pin_memory)

# ======================================================================================================================

base_beta_decay = tt.functional.halflife_to_decay(784 / (kernel_size ** 2))
print(f"Base Beta: {base_beta_decay}")

hidden_features = 256
model_template = snn.Sequential(
	nn.Linear(kernel_size ** 2, hidden_features),
	snn.LIF(hidden_features, base_beta_decay),
	nn.Linear(hidden_features, hidden_features),
	snn.LIF(hidden_features, base_beta_decay),
	nn.Linear(hidden_features, hidden_features),
	snn.LIF(hidden_features, base_beta_decay),
	nn.Linear(hidden_features, 10),
	snn.Readout(10, base_beta_decay),
	nn.Softmax(-1)
).to(device=device)

num_epochs = 10

total_params = sum(p.numel() for p in model_template.parameters())
print(f"Total parameters: {total_params:,}")
time.sleep(0.1)

loss_fn = tt.loss.soft_cross_entropy
# loss_fn = nn.functional.mse_loss

train_types = ["bptt", "online"]

for train_type in train_types:
	model = copy.deepcopy(model_template)
	optimizer = torch.optim.AdamW(model.parameters(), 1e-3 if train_type == "bptt" else 1e-5)

	loss_manager = tt.plot.MeasurementManager(f"{train_type} - Loss", [0.0, 0.9, 0.99])
	accuracy_manager = tt.plot.MeasurementManager(f"{train_type} - Accuracy", [0.0, 0.9, 0.99])

	# TRAIN
	model.train()
	for e in range(num_epochs):
		for (img, seq, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"{train_type}, E{e}-TRAIN"):
			img, seq, label = img.to(device), torch.bernoulli(seq).to(device), label.to(device)
			t, b, snn = seq.shape

			model.zero_states()
			model.zero_grad()

			if train_type == "bptt":
				for view in seq:
					model_output = model(view)
				loss = loss_fn(model_output, label)
				loss.backward()

			if train_type == "online":
				for index, view in enumerate(seq):
					if index < t - 1:
						with torch.no_grad():
							model_output = model(view)
					else:
						model_output = model(view)
				loss = loss_fn(model_output, label)
				loss.backward()

			optimizer.step()

			loss_manager.append(loss.item())
			pred_classes = model_output.argmax(dim=1)
			true_classes = label.argmax(dim=1)
			frac_correct = (pred_classes == true_classes).sum().item() / b
			accuracy_manager.append(frac_correct)

	loss_manager.plot()
	accuracy_manager.plot()

	# TEST
	model.eval()
	test_loss, test_accuracy = 0, 0
	for (img, seq, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"{train_type} - TEST"):
		with torch.no_grad():
			img, seq, label = img.to(device), torch.bernoulli(seq).to(device), label.to(device)
			t, b, snn = seq.shape

			model.zero_states()

			for view in seq:
				model_output = model(view)
			loss = loss_fn(model_output, label)

			test_loss += loss.item()
			true_classes = label.argmax(dim=1)
			pred_classes = model_output.argmax(dim=1)
			num_correct = (pred_classes == true_classes).sum().item()
			test_accuracy += num_correct

	test_loss /= len(test_dataloader)
	test_accuracy /= len(test_dataset)

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
