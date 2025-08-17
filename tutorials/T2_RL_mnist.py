import torch
import tracetorch as tt
from tracetorch import snn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

min_prob = 0.1
max_prob = 0.5

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

REFLECT = tt.loss.Reflect(num_in=model.num_out, decay=0.9)

model_optimizer = torch.optim.AdamW(params=model.get_learnable_parameters(), lr=1e-4)
reflect_optimizer = torch.optim.AdamW(params=REFLECT.get_learnable_parameters(), lr=1e-5)

loss_manager = tt.plot.MeasurementManager(title="Loss")
accuracy_manager = tt.plot.MeasurementManager(title="Accuracy")
reflect_manager = tt.plot.MeasurementManager(title="REFLECT decay", decay=torch.tensor([0.0]))

from tqdm import tqdm

think_steps = 20
num_epochs = 1

for epoch in range(num_epochs):
	for index, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True, desc=f"E{epoch}"):
		x, y = x.to(device), y.to(device)
		model.zero_states()
		REFLECT.zero_states(clear_reward=False)

		aggregate = torch.zeros(model.num_out).to(device)
		for _ in range(think_steps):
			bern = torch.bernoulli(x)
			model_dist = model.forward(bern)
			aggregate += model_dist
			model_output = tt.functional.sample_softmax(model_dist)
			REFLECT.forward(model_dist, model_output)
		aggregate /= think_steps
		reward = 1 if aggregate.argmax().item() == y.argmax().item() else 0

		loss, ls = REFLECT.backward(reward)
		# print(f"Average Distribution: {average_distribution}")
		# print(f"Passed LS max: {ls.abs().max()}")
		model.backward(ls)

		model_optimizer.step()
		reflect_optimizer.step()
		model.zero_grad(set_to_none=True)
		REFLECT.zero_grad(set_to_none=True)

		loss_manager.append(loss)
		accuracy_manager.append(reward)
		reflect_manager.append(torch.nn.functional.sigmoid(REFLECT.decay))

		if (index % 10000 == 0) and (index != 0):
			loss_manager.plot()
			accuracy_manager.plot()
			reflect_manager.plot()

loss_manager.plot()
accuracy_manager.plot()
reflect_manager.plot()

net_loss = torch.tensor(0.).to(device)
net_accuracy = 0
mistakes_matrix = torch.zeros(10, 10)

for index, (x, y) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=True, desc=f"Testing"):
	x, y = x.to(device), y.to(device)
	model.zero_states()
	aggregate = torch.zeros(model.num_out).to(device)
	raw_image = torch.zeros_like(x)
	model_dists = []
	for _ in range(think_steps):
		bern = torch.bernoulli(x)
		raw_image += bern
		model_dist = model.forward(bern)
		aggregate += model_dist
		model_dists.append(model_dist)
	aggregate /= think_steps
	raw_image /= think_steps
	if index % 1000 == 0:
		tt.plot.render_image(raw_image.unsqueeze(0).view(28, 28))
		tt.plot.spike_train(model_dists, title="Model outputs")
	loss, ls = tt.loss.mse(aggregate, y)
	net_loss += loss
	chosen_index = int(aggregate.argmax().item())
	real_index = int(y.argmax().item())
	correct = chosen_index == real_index
	if not correct:
		mistakes_matrix[real_index][chosen_index] += 1
	else:
		net_accuracy += 1

net_loss /= len(test_dataloader)
net_accuracy /= len(test_dataloader)

print(f"\tTEST\nLoss: {net_loss}\nAccuracy: {net_accuracy}")
for index, mistakes in enumerate(mistakes_matrix):
	print(f"Index: {index} confused for: {mistakes}")

in_trace_decays = [torch.nn.functional.sigmoid(layer.in_trace_decay) for layer in model.layers]
mem_decays = [torch.nn.functional.sigmoid(layer.mem_decay) for layer in model.layers]
weights = [layer.weight for layer in model.layers]
thresholds = [torch.nn.functional.softplus(layer.threshold) for layer in model.layers[:-1]]

tt.plot.distributions(in_trace_decays, title="Input trace decay")
tt.plot.distributions(mem_decays, title="Membrane decay")
tt.plot.distributions(weights, title="Weights")
tt.plot.distributions(thresholds, title="Thresholds")

tt.plot.render_image(mistakes_matrix)
