# ======================================================================================================================


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

noise_offset = 0.1
min_prob = 0.0
max_prob = (1.0 / num_timesteps)

# min_prob, max_prob, noise_offset = 0.0, 0.05, 0.01

train_dataset = OneHotMNIST(train=True, min_val=min_prob, max_val=max_prob, offset=noise_offset)
test_dataset = OneHotMNIST(train=False, min_val=min_prob, max_val=max_prob, offset=noise_offset)

batch_size = 32
num_to_visualize = 5

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class SNN(snn.TTModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 5, dilation=2),
            # kernel 5x5 @ dilation 2 becomes 9x9, 28-9+1=20
            snn.LIF(8, torch.rand(8), 1.0, dim=-3),
            nn.Conv2d(8, 16, 5, dilation=2),
            # kernel 5x5 @ dilation 2 becomes 9x9, 20-9+1 = 12
            snn.LIF(16, torch.rand(16), 1.0, dim=-3),
            nn.Conv2d(16, 16, 5, dilation=2),
            # kernel 5x5 @ dilation 2 becomes 9x9, 12-9+1 = 4
            nn.Flatten(),
            snn.Readout(256, torch.rand(256)),  # 4x4x16=256
            nn.Linear(256, 10),
            nn.Softmax(-1),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):
        return self.net(x)


model = SNN().to(device)
print(f"\nNum params: {model.get_param_count():,}")
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)

loss_fn = tt.loss.soft_cross_entropy
# loss_fn = nn.functional.mse_loss

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
        true_classes = label.argmax(dim=1)
        frac_correct = (pred_classes == true_classes).sum().item() / batch_size

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
            true_classes = label.argmax(dim=1)
            frac_correct = (pred_classes == true_classes).sum().item() / batch_size

            test_acc += frac_correct

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        print(f"TEST - Loss: {test_loss} | Acc: {test_acc}")
