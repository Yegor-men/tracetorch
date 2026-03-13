import torch
from torch import nn
import tracetorch as tt
from tracetorch import snn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# ======================================================================================================================

class MNIST(torch.utils.data.Dataset):
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
        return image, label

    def __len__(self):
        return len(self.dataset)


# ======================================================================================================================

num_epochs = 5
num_timesteps = 20

min_prob, max_prob, noise_offset = 0.0, 0.90, 0.10

train_dataset = MNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = MNIST(train=False, min_val=0.0, max_val=1.0, offset=0.0)

batch_size = 100
num_to_visualize = 5

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


def spike_fn(x):
    return nn.functional.sigmoid(2 * x)


deterministic = True


class SNN(snn.TTModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28+2-2=28
            snn.LIB(16, dim=-3, spike_fn=spike_fn, deterministic=deterministic),
            nn.MaxPool2d(2, 2),  # 28/2=14
            nn.Conv2d(16, 32, 3, padding=1),  # 14+2-2=14
            snn.LIB(32, dim=-3, spike_fn=spike_fn, deterministic=deterministic),
            nn.MaxPool2d(2, 2),  # 14/2=7
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            snn.LI(128),
            nn.Linear(128, 10),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


model = SNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
snn_params = model.get_param_count()
print(f"Total: {total_params:,} -> SNN: {snn_params:,} | Non-SNN: {total_params - snn_params:,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

loss_fn = nn.functional.cross_entropy

train_losses, train_accs = [], []

for e in range(num_epochs):
    model.train()

    for (image, label) in tqdm(train_dataloader, total=len(train_dataloader), desc=f"TRAIN - E{e}"):
        image, label = image.to(device), label.to(device)

        model.zero_grad()
        model.zero_states()

        running_loss = 0.0
        for t in range(num_timesteps):
            model_output = model(torch.bernoulli(image))
            loss = loss_fn(model_output, label)
            running_loss = running_loss + loss
        running_loss = running_loss / num_timesteps

        pred_classes = model_output.argmax(dim=1)
        frac_correct = (pred_classes == label).sum().item() / batch_size

        train_losses.append(running_loss.item())
        train_accs.append(frac_correct)

        running_loss.backward()
        optimizer.step()

    plt.title("LOSS")
    plt.plot(train_losses, label="train")
    plt.legend()
    plt.show()

    plt.title("ACC")
    plt.plot(train_accs, label="train")
    plt.legend()
    plt.show()

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (image, label) in tqdm(test_dataloader, total=len(test_dataloader), desc=f"TEST - E{e}"):
            image, label = image.to(device), label.to(device)

            model.zero_states()

            running_loss = 0.0
            for t in range(num_timesteps):
                model_output = model(torch.bernoulli(image))
                loss = loss_fn(model_output, label)
                running_loss = running_loss + loss
            running_loss = running_loss / num_timesteps

            test_loss += running_loss.item()

            pred_classes = model_output.argmax(dim=1)
            frac_correct = (pred_classes == label).sum().item() / batch_size

            test_acc += frac_correct

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        print(f"TEST - Loss: {test_loss} | Acc: {test_acc}")

with torch.no_grad():
    image_batch, label_batch = next(iter(train_dataloader))
    image_batch, label_batch = image_batch.to(device), label_batch.to(device)
    for i in range(10):
        image, label = image_batch[i].unsqueeze(0), label_batch[i].unsqueeze(0)

        model.zero_states()
        empty_image = torch.zeros_like(image)
        model_outputs = []
        losses = []
        running_loss = 0.0
        for t in range(20):
            spk_image = torch.bernoulli(image)
            empty_image += spk_image
            model_output = model(spk_image).squeeze(0)
            model_outputs.append(nn.functional.softmax(model_output, dim=-1))
            loss = loss_fn(model_output, label)
            running_loss += loss.item() / num_timesteps
            losses.append(loss.item())

        tt.plot.render_image(empty_image, title=f"Loss: {running_loss:.3f}")
        tt.plot.spike_train(model_outputs, title="Spike Train")
        plt.title("Loss over time")
        plt.plot(losses)
        plt.show()
