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

device = "cuda" if torch.cuda.is_available() else "cpu"

num_hidden = 32

pytorch_model = nn.Sequential(
	nn.LazyLinear(out_features=num_hidden),
	nn.LeakyReLU(),
	nn.LazyLinear(out_features=num_hidden),
	nn.LeakyReLU(),
	nn.LazyLinear(out_features=10),
	nn.Softmax(dim=-1),
).to(device)

decay = tt.functional.halflife_to_decay(3)
tracetorch_model = snn.Sequential(
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

lr = 1e-4
pytorch_model_optimizer = torch.optim.AdamW(pytorch_model.parameters(), lr=lr)
tracetorch_model_optimizer = torch.optim.AdamW(tracetorch_model.parameters(), lr=lr)

pytorch_model_accuracy = tt.plot.MeasurementManager(title="PyTorch Model Accuracy")
tracetorch_model_accuracy = tt.plot.MeasurementManager(title="traceTorch Model Accuracy")

num_epochs = 1
think_steps = 10

for epoch in range(num_epochs):
	for _, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{epoch}"):
		x, y = x.to(device), y.to(device)
		correct_class = y.argmax().item()

		pytorch_out = pytorch_model.forward(x)
		pytorch_model_accuracy.append(1 if pytorch_out.argmax().item() == correct_class else 0)
		pytorch_loss = torch.nn.functional.cross_entropy(pytorch_out, y)
		pytorch_loss.backward()

		tracetorch_model.zero_states()
		aggregate = torch.zeros(tracetorch_model.num_out).to(device)
		for t in range(think_steps):
			tracetorch_out = tracetorch_model.forward(x)
			aggregate += tracetorch_out
			tracetorch_loss, ls = tt.loss.cross_entropy(tracetorch_out, y)
			tracetorch_model.backward(ls)
		aggregate /= think_steps
		tracetorch_model_accuracy.append(1 if aggregate.argmax().item() == correct_class else 0)

		pytorch_model_optimizer.step()
		pytorch_model.zero_grad()

		tracetorch_model.elig_to_grad()
		tracetorch_model_optimizer.step()
		tracetorch_model.zero_grad()

		if (_ + 1) % 10_000 == 0:
			pytorch_model_accuracy.plot()
			tracetorch_model_accuracy.plot()

pytorch_model_accuracy.plot()
tracetorch_model_accuracy.plot()

pytorch_correct = 0
tracetorch_correct = 0

for _, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True, desc=f"Testing"):
	with torch.no_grad():
		x, y = x.to(device), y.to(device)
		correct_class = y.argmax().item()

		pytorch_out = pytorch_model.forward(x)
		pytorch_correct += 1 if pytorch_out.argmax().item() == correct_class else 0

		tracetorch_model.zero_states()
		aggregate = torch.zeros(tracetorch_model.num_out).to(device)
		for t in range(think_steps):
			tracetorch_out = tracetorch_model.forward(x)
			aggregate += tracetorch_out
		aggregate /= think_steps
		tracetorch_correct += 1 if aggregate.argmax().item() == correct_class else 0

pytorch_correct /= len(test_dataloader)
tracetorch_correct /= len(test_dataloader)

print(f"PyTorch % accuracy: {pytorch_correct * 100:.2f}")
print(f"traceTorch % accuracy: {tracetorch_correct * 100:.2f}")
